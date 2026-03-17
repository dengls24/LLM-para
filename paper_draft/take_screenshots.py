"""
Take website screenshots for the paper Figure 1 (system interface).
Captures multiple tabs of the LLM-Para tool with data loaded.
"""

import asyncio, os, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs('figures', exist_ok=True)

LLAMA3_CONFIG = {
    "hidden_size": 4096, "num_heads": 32, "num_key_value_heads": 8,
    "num_layers": 32, "intermediate_size": 14336, "vocab_size": 128256,
    "seq_len": 2048, "batch_size": 1, "max_gen_len": 4096,
    "use_gate_ffn": True, "use_rmsnorm": True,
    "rope_theta": 500000.0, "rope_scaling_factor": 1.0,
    "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                     "kv_cache": 16, "rope_bit": 32},
    "hardware_key": "NVIDIA H100 SXM"
}

async def take_screenshots():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": 1600, "height": 900},
            device_scale_factor=2   # retina 2x for crisp screenshots
        )
        page = await ctx.new_page()

        print("  Opening LLM-Para...")
        await page.goto("http://localhost:5000", wait_until="networkidle")
        await page.wait_for_timeout(1500)

        # ── 1. Trigger analysis with LLaMA-3 8B config ──────────────────────
        print("  Loading LLaMA-3 8B preset...")
        await page.select_option('#modelPreset', label='LLaMA-3 8B')
        await page.wait_for_timeout(800)

        # Select H100 hardware
        try:
            await page.select_option('#hardwarePreset', value='NVIDIA H100 SXM')
            await page.wait_for_timeout(400)
        except:
            pass

        # Click Analyze
        print("  Running analysis...")
        await page.click('#analyzeBtn')
        await page.wait_for_timeout(3000)

        # ── Screenshot A: Operations Table ───────────────────────────────────
        print("  Screenshot: Operations Table...")
        try:
            tab_table = await page.query_selector('[data-tab="table"]')
            if tab_table: await tab_table.click()
            await page.wait_for_timeout(800)
        except:
            pass
        await page.screenshot(path='figures/screen_table.png',
                               full_page=False, clip={'x':0,'y':60,'width':1600,'height':840})

        # ── Screenshot B: Roofline Tab ────────────────────────────────────────
        print("  Screenshot: Roofline...")
        try:
            tab_roof = await page.query_selector('[data-tab="roofline"]')
            if tab_roof: await tab_roof.click()
            # select H100
            await page.select_option('#rooflineHW', value='NVIDIA H100 SXM')
            await page.wait_for_timeout(2000)
        except Exception as e:
            print(f"    (roofline tab: {e})")
        await page.screenshot(path='figures/screen_roofline.png',
                               full_page=False, clip={'x':0,'y':60,'width':1600,'height':840})

        # ── Screenshot C: Energy Tab ──────────────────────────────────────────
        print("  Screenshot: Energy Roofline...")
        try:
            tab_e = await page.query_selector('[data-tab="energy"]')
            if tab_e: await tab_e.click()
            await page.wait_for_timeout(400)
            await page.select_option('#energyHW', value='NVIDIA H100 SXM')
            await page.click('#runEnergyBtn')
            await page.wait_for_timeout(3000)
        except Exception as e:
            print(f"    (energy tab: {e})")
        await page.screenshot(path='figures/screen_energy.png',
                               full_page=False, clip={'x':0,'y':60,'width':1600,'height':840})

        # ── Screenshot D: DSE Tab ─────────────────────────────────────────────
        print("  Screenshot: DSE Explorer...")
        try:
            tab_dse = await page.query_selector('[data-tab="dse"]')
            if tab_dse: await tab_dse.click()
            await page.wait_for_timeout(400)
            await page.click('#runDSEBtn')
            await page.wait_for_timeout(8000)
        except Exception as e:
            print(f"    (dse tab: {e})")
        await page.screenshot(path='figures/screen_dse.png',
                               full_page=False, clip={'x':0,'y':60,'width':1600,'height':840})

        # ── Screenshot E: Hetero Tab ──────────────────────────────────────────
        print("  Screenshot: Hetero Architecture...")
        try:
            tab_h = await page.query_selector('[data-tab="hetero"]')
            if tab_h: await tab_h.click()
            await page.wait_for_timeout(400)
            # select first hetero hw
            hw_sel = await page.query_selector('#heteroHW')
            options = await hw_sel.query_selector_all('option')
            for opt in options:
                val = await opt.get_attribute('value')
                if val:
                    await page.select_option('#heteroHW', value=val)
                    break
            await page.click('#runHeteroBtn')
            await page.wait_for_timeout(4000)
        except Exception as e:
            print(f"    (hetero tab: {e})")
        await page.screenshot(path='figures/screen_hetero.png',
                               full_page=False, clip={'x':0,'y':60,'width':1600,'height':840})

        # ── Screenshot F: Full-page home (for paper system figure) ───────────
        print("  Screenshot: Full interface (for Fig.1)...")
        await page.goto("http://localhost:5000", wait_until="networkidle")
        await page.wait_for_timeout(1500)
        await page.select_option('#modelPreset', label='LLaMA-3 8B')
        await page.wait_for_timeout(600)
        try:
            await page.select_option('#hardwarePreset', value='NVIDIA H100 SXM')
        except: pass
        await page.click('#analyzeBtn')
        await page.wait_for_timeout(3500)

        # Capture the full results view at slightly smaller viewport for paper
        await page.set_viewport_size({"width": 1400, "height": 860})
        await page.wait_for_timeout(500)
        await page.screenshot(path='figures/fig_system_screenshot.png',
                               full_page=False)

        await browser.close()
        print("\n  All screenshots saved to paper_draft/figures/")
        for f in sorted(os.listdir('figures')):
            if f.startswith('screen') or f == 'fig_system_screenshot.png':
                size = os.path.getsize(f'figures/{f}')
                print(f"    figures/{f}  ({size//1024} KB)")

asyncio.run(take_screenshots())
