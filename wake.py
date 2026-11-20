import os
import importlib

try:
    sync_playwright = importlib.import_module("playwright.sync_api").sync_playwright
except ModuleNotFoundError as error:
    raise SystemExit(
        "Playwright is required. Install it with: "
        "pip install playwright && playwright install chromium"
    ) from error

URL = os.environ.get(
    "MOVIELENS_URL", "https://movielens-hybrid-recommender1111.streamlit.app/"
)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)

    # Sleep ma xa vane wake-up button dekhincha
    try:
        button = page.get_by_role("button", name="Yes, get this app back up!")
        button.wait_for(timeout=10000)
        button.click()
        print("App was asleep. Wake-up button clicked.")
        page.wait_for_timeout(30000)  # app uthna time dine
    except Exception:
        print("App was already awake.")

    browser.close()
