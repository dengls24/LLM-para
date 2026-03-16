"""
LLM-Para Web Server
Flask backend providing REST API for LLM analysis and visualization.
"""

import json
import csv
import io
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

from analyzer import LLMAnalyzer
from configs import MODEL_CONFIGS, HARDWARE_CONFIGS, CATEGORY_COLORS, PHASE_SHAPES

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)


# ─── Page Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ─── API: Config endpoints ────────────────────────────────────────────────────

@app.route('/api/models', methods=['GET'])
def get_models():
    """Return list of preset model configurations."""
    models = []
    for name, cfg in MODEL_CONFIGS.items():
        models.append({
            'name': name,
            'hidden_size': cfg['hidden_size'],
            'num_heads': cfg['num_heads'],
            'num_layers': cfg['num_layers'],
            'has_gqa': cfg.get('num_key_value_heads', cfg['num_heads']) != cfg['num_heads'],
            'has_moe': bool(cfg.get('num_experts_per_tok')),
            'has_rope': bool(cfg.get('rope_theta')),
            'has_mla': bool(cfg.get('use_mla')),
            'config': cfg,
        })
    return jsonify({'models': models})


@app.route('/api/hardware', methods=['GET'])
def get_hardware():
    """Return list of preset hardware configurations."""
    hw_list = []
    for key, hw in HARDWARE_CONFIGS.items():
        hw_list.append({
            'key': key,
            'name': hw['name'],
            'category': hw['category'],
            'peak_performance': hw['peak_performance'],
            'peak_performance_fp16': hw.get('peak_performance_fp16', hw['peak_performance']),
            'memory_bandwidth': hw['memory_bandwidth'],
            'memory_capacity': hw.get('memory_capacity', 0),
        })
    return jsonify({'hardware': hw_list})


@app.route('/api/constants', methods=['GET'])
def get_constants():
    """Return frontend constants (colors, shapes, etc.)."""
    return jsonify({
        'category_colors': CATEGORY_COLORS,
        'phase_shapes': PHASE_SHAPES,
    })


# ─── API: Analysis ────────────────────────────────────────────────────────────

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Run LLM analysis with given configuration.
    Accepts JSON body with model config.
    Returns per-operator results, summary, and roofline data.
    """
    try:
        cfg = request.get_json()
        if not cfg:
            return jsonify({'error': 'No configuration provided'}), 400

        # Validate required fields
        required = ['hidden_size', 'num_heads', 'num_layers', 'seq_len',
                    'batch_size', 'quant_config']
        for field in required:
            if field not in cfg:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        analyzer = LLMAnalyzer(cfg)
        results = analyzer.analyze()
        summary = analyzer.get_summary(results)

        # Get roofline data for selected hardware (if provided)
        roofline_data = None
        hw_key = cfg.get('hardware_key')
        if hw_key and hw_key in HARDWARE_CONFIGS:
            roofline_data = analyzer.get_roofline_data(results, HARDWARE_CONFIGS[hw_key])

        # Serialize results (convert floats for JSON)
        def safe(v):
            if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')):
                return 0
            return v

        clean_results = []
        for r in results:
            clean_results.append({k: safe(v) for k, v in r.items()})

        return jsonify({
            'success': True,
            'results': clean_results,
            'summary': summary,
            'roofline': roofline_data,
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/roofline', methods=['POST'])
def roofline():
    """
    Get roofline analysis for given results + hardware config.
    """
    try:
        body = request.get_json()
        results = body.get('results', [])
        hw_key = body.get('hardware_key')

        if not hw_key or hw_key not in HARDWARE_CONFIGS:
            return jsonify({'error': f'Unknown hardware: {hw_key}'}), 400

        # Need a dummy analyzer just for roofline computation
        # Use a minimal config
        cfg = body.get('config', {'hidden_size': 4096, 'num_heads': 32,
                                   'num_layers': 32, 'seq_len': 2048,
                                   'batch_size': 1, 'quant_config': {
                                       'activation': 16, 'weight_attn': 16,
                                       'weight_ffn': 16, 'kv_cache': 16}})
        analyzer = LLMAnalyzer(cfg)
        roofline_data = analyzer.get_roofline_data(results, HARDWARE_CONFIGS[hw_key])
        return jsonify({'success': True, 'roofline': roofline_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── API: Export ──────────────────────────────────────────────────────────────

@app.route('/api/export/csv', methods=['POST'])
def export_csv():
    """Export analysis results as CSV."""
    try:
        body = request.get_json()
        results = body.get('results', [])
        if not results:
            return jsonify({'error': 'No results to export'}), 400

        output = io.StringIO()
        if results:
            writer = csv.DictWriter(output, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='llm_para_analysis.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/json', methods=['POST'])
def export_json():
    """Export analysis results as JSON."""
    try:
        body = request.get_json()
        results = body.get('results', [])
        summary = body.get('summary', {})
        output = json.dumps({'results': results, 'summary': summary}, indent=2)
        return send_file(
            io.BytesIO(output.encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name='llm_para_analysis.json'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── API: Compare multiple configs ────────────────────────────────────────────

@app.route('/api/compare', methods=['POST'])
def compare():
    """
    Compare multiple model configurations.
    Accepts list of configs, returns summaries for all.
    """
    try:
        body = request.get_json()
        configs = body.get('configs', [])
        if not configs:
            return jsonify({'error': 'No configs to compare'}), 400

        comparisons = []
        for cfg in configs:
            analyzer = LLMAnalyzer(cfg)
            results = analyzer.analyze()
            summary = analyzer.get_summary(results)
            comparisons.append({
                'name': cfg.get('name', 'Custom'),
                'summary': summary,
            })

        return jsonify({'success': True, 'comparisons': comparisons})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  LLM-Para Web Server")
    print("  Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
