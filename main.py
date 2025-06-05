from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import os
import re
import requests
import extruct
from w3lib.html import get_base_url
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
import json
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Web Audit Data Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
DATA_FILE_PATH = BASE_DIR / "Data" / "efax_internal_html.csv"
ALT_TAG_DATA_PATH = BASE_DIR / "Data" / "images_missing_alt_text_efax.csv"
ORPHAN_PAGES_DATA_PATH = BASE_DIR / "Data" / "efax_orphan_urls.csv"

SCHEMA_CHECKLIST = [
    ("Breadcrumbs", "BreadcrumbList"),
    ("FAQ", "FAQPage"),
    ("Article", "Article"),
    ("Video", "VideoObject"),
    ("Organization", "Organization"),
    ("How-to", "HowTo"),
    ("WebPage", "WebPage"),
    ("Product", "Product"),
    ("Review", "Review"),
    ("Person", "Person"),
    ("Event", "Event"),
    ("Recipe", "Recipe"),
    ("LocalBusiness", "LocalBusiness"),
    ("CreativeWork", "CreativeWork"),
    ("ItemList", "ItemList"),
    ("JobPosting", "JobPosting"),
    ("Course", "Course"),
    ("ImageObject", "ImageObject"),
    ("Service", "Service"),
]

URL_VARIATIONS = [
    "",
    "/about",
    "/about-us",
    "/products",
    "/services",
    "/solutions",
    "/blog",
    "/blogs",
    "/cyberglossary",
    "/news",
    "/resources",
    "/how-to",
    "/tutorials",
    "/guides",
    "/faq",
    "/help",
    "/support",
    "/contact",
    "/contact-us",
    "/reviews",
    "/testimonials",
    "/portfolio",
    "/cases",
    "/case-studies",
    "/team",
    "/careers",
    "/jobs",
    "/catalog",
    "/pricing",
    "/plans",
    "/login",
    "/signup",
    "/register",
]

# Pydantic models for request/response
class FileUploadResponse(BaseModel):
    message: str
    rows: int
    columns: List[str]

class AnalysisResponse(BaseModel):
    domain: str
    report: List[Dict[str, Any]]
    detailed_data: Dict[str, Any]
    alt_tag_data: Optional[List[Dict[str, Any]]] = None
    orphan_pages_data: Optional[List[Dict[str, Any]]] = None

class FileValidationResponse(BaseModel):
    files_exist: bool
    missing_files: List[str]

# Helper functions (same as original)
def is_valid_page_url(url):
    if re.search(r'\.(jpg|jpeg|png|gif|bmp|pdf|doc|docx|xls|xlsx|css|js)$', url, re.IGNORECASE):
        return False
    if "wp-content" in url.lower() or "wp-uploads" in url.lower():
        return False
    return True

def extract_schemas(url, timeout=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        base_url = get_base_url(response.text, response.url)

        data = extruct.extract(
            response.text, base_url=base_url, syntaxes=["json-ld", "microdata", "rdfa"]
        )
        return data
    except requests.exceptions.RequestException:
        return None

def flatten_schema(schema_item):
    if isinstance(schema_item, list):
        for item in schema_item:
            yield from flatten_schema(item)
    elif isinstance(schema_item, dict):
        if "@graph" in schema_item:
            yield from flatten_schema(schema_item["@graph"])
        yield schema_item

def extract_schema_names(schemas):
    schema_names = set()

    for item in flatten_schema(schemas.get("json-ld", [])):
        if "@type" in item:
            if isinstance(item["@type"], list):
                for t in item["@type"]:
                    schema_names.add(t)
            else:
                schema_names.add(item["@type"])

    for item in flatten_schema(schemas.get("microdata", [])):
        if "type" in item:
            if isinstance(item["type"], list):
                for t in item["type"]:
                    schema_names.add(t)
            else:
                schema_names.add(item["type"])

    for item in flatten_schema(schemas.get("rdfa", [])):
        if "type" in item:
            if isinstance(item["type"], list):
                for t in item["type"]:
                    schema_names.add(t)
            else:
                schema_names.add(item["type"])

    return sorted(schema_names)

def normalize_base_url(url):
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

def get_domain_from_df(df):
    """Extract domain from the first URL in the dataframe"""
    if df is not None and len(df) > 0 and "Address" in df.columns:
        first_url = df["Address"].iloc[0]
        if "://" in first_url:
            return first_url.split("/")[2]
        else:
            return first_url.split("/")[0]
    return None

def check_schema_markup(domain, max_workers=10, timeout=8):
    base_url = normalize_base_url(domain)
    all_schemas = set()
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for variation in URL_VARIATIONS:
                full_url = urljoin(base_url, variation)
                future = executor.submit(extract_schemas, full_url, timeout)
                futures.append((future, full_url))
            
            for future, url in futures:
                try:
                    schemas = future.result(timeout=timeout + 2) 
                    if schemas:
                        schema_names = extract_schema_names(schemas)
                        all_schemas.update(schema_names)
                except Exception as e:
                    continue
                    
        return all_schemas
    except Exception as e:
        print(f"Error in check_schema_markup: {str(e)}")
        return set()

def update_schema_markup_analysis(df, report, expected_outcomes, sources):
    domain = get_domain_from_df(df)
    
    if domain:
        try:
            found_schemas = check_schema_markup(domain)
            
            if found_schemas:
                schema_list = sorted(list(found_schemas))
                current_value = f"Found {len(schema_list)} types: {', '.join(schema_list)}"

                if len(schema_list) >= 5:
                    status = "✅ Pass"
                elif len(schema_list) >= 2:
                    status = "ℹ️ Review"
                else:
                    status = "❌ Fail"
            else:
                current_value = "No schema markup detected"
                status = "❌ Fail"
                
        except Exception as e:
            current_value = f"Error checking schemas: {str(e)}"
            status = "ℹ️ Not Available"
            print(f"Schema analysis error: {str(e)}")
    else:
        current_value = "Cannot extract domain from data"
        status = "ℹ️ Not Available"
    
    report["Category"].append("Metadata & Schema")
    report["Parameters"].append("Schema Markup")
    report["Current Value"].append(current_value)
    report["Expected Value"].append(expected_outcomes["Schema Markup"])
    report["Source"].append(sources["Schema Markup"])
    report["Status"].append(status)

def analyze_screaming_frog_data(df, alt_tag_df=None, orphan_pages_df=None):
    expected_outcomes = {
        "Website performance on desktop": "Score > 90",
        "Website performance on mobile": "Score > 80",
        "Core Web Vitals on desktop": "Pass",
        "Core Web Vitals on mobile": "Pass",
        "Accessibility Score": "Score > 90",
        "SEO Score": "Score > 90",
        "Mobile friendliness": "Pass",
        "Indexed pages": "All active pages are indexed",
        "Non indexed pages": "No active pages are in no index state.\nMinimal or no indexed pages",
        "Robots.txt file optimization": "Optimized",
        "Sitemap file optimization": "All active website URLs are part of sitemap",
        "Broken internal links (404)": "0 broken links",
        "Broken external links": "0 broken links",
        "Broken backlinks": "0 broken backlinks",
        "Broken Images": "0 broken Images",
        "Orphan page": "No orphan page",
        "Canonical Errors": "No page with canonical error",
        "Information architecture": "Site structure & navigation is well defined & easy to understand",
        "Header tags structure": "Content structure is well-defined and easy to understand",
        "Backlinks": "No of backlinks",
        "Domain authority": "DA >70",
        "Spam Score": "Score <5",
        "Duplicate content": "Minimal or no pages with issues",
        "Img alt tag": "Minimal or no pages with issues",
        "Duplicate & missing H1": "Minimal or no pages with issues",
        "Duplicate & missing meta title": "Minimal or no pages with issues",
        "Duplicate & missing description": "Minimal or no pages with issues",
        "Schema Markup": "Schema implementation opportunities",
    }

    sources = {
        "Website performance on desktop": "Pagespeedinsights",
        "Website performance on mobile": "Pagespeedinsights",
        "Core Web Vitals on desktop": "Pagespeedinsights",
        "Core Web Vitals on mobile": "Pagespeedinsights",
        "Accessibility Score": "Pagespeedinsights",
        "SEO Score": "Pagespeedinsights",
        "Mobile friendliness": "Manual",
        "Indexed pages": "Google search console",
        "Non indexed pages": "Google search console",
        "Robots.txt file optimization": "Manual",
        "Sitemap file optimization": "Manual",
        "Broken internal links (404)": "Ahrefs",
        "Broken external links": "Ahrefs",
        "Broken backlinks": "Ahrefs",
        "Broken Images": "Manual",
        "Orphan page": "Screamfrog",
        "Canonical Errors": "Screamfrog",
        "Information architecture": "Ahrefs",
        "Header tags structure": "Manual",
        "Backlinks": "Ahrefs",
        "Domain authority": "Moz",
        "Spam Score": "Moz",
        "Duplicate content": "Screamfrog",
        "Img alt tag": "Screamfrog",
        "Duplicate & missing H1": "Screamfrog",
        "Duplicate & missing meta title": "Screamfrog",
        "Duplicate & missing description": "Screamfrog",
        "Schema Markup": "Automated Schema Detection"
    }

    report = {
        "Category": [],
        "Parameters": [],
        "Current Value": [],
        "Expected Value": [],
        "Source": [],
        "Status": [],
    }

    performance_metrics = [
        "Website performance on desktop",
        "Website performance on mobile",
        "Core Web Vitals on desktop",
        "Core Web Vitals on mobile",
        "Accessibility Score",
        "SEO Score",
        "Mobile friendliness",
    ]

    for metric in performance_metrics:
        report["Category"].append("Performance & Core Web Vitals")
        report["Parameters"].append(metric)
        report["Current Value"].append("N/A")
        report["Expected Value"].append(expected_outcomes[metric])
        report["Source"].append(sources[metric])
        report["Status"].append("ℹ️ Not Available")

    if "Indexability" in df.columns and "Indexability Status" in df.columns:
        indexed_pages = len(df[df["Indexability"] == "Indexable"])
        report["Category"].append("Crawling & Indexing")
        report["Parameters"].append("Indexed pages")
        report["Current Value"].append(indexed_pages)
        report["Expected Value"].append(expected_outcomes["Indexed pages"])
        report["Source"].append(sources["Indexed pages"])
        report["Status"].append("ℹ️ Review")

        non_indexed_pages = len(
            df[df["Indexability Status"].str.contains("noindex", na=False)]
        )
        report["Category"].append("Crawling & Indexing")
        report["Parameters"].append("Non indexed pages")
        report["Current Value"].append(non_indexed_pages)
        report["Expected Value"].append(expected_outcomes["Non indexed pages"])
        report["Source"].append(sources["Non indexed pages"])
        report["Status"].append("ℹ️ Review" if non_indexed_pages > 0 else "✅ Pass")
    else:
        for param in ["Indexed pages", "Non indexed pages"]:
            report["Category"].append("Crawling & Indexing")
            report["Parameters"].append(param)
            report["Current Value"].append("N/A")
            report["Expected Value"].append(expected_outcomes[param])
            report["Source"].append(sources[param])
            report["Status"].append("ℹ️ Not Available")

    for param in ["Robots.txt file optimization", "Sitemap file optimization"]:
        report["Category"].append("Crawling & Indexing")
        report["Parameters"].append(param)
        report["Current Value"].append("N/A")
        report["Expected Value"].append(expected_outcomes[param])
        report["Source"].append(sources[param])
        report["Status"].append("ℹ️ Not Available")

    broken_internal_links = len(df[df["Status Code"] == 404])
    report["Category"].append("Site Health & Structure")
    report["Parameters"].append("Broken internal links (404)")
    report["Current Value"].append(broken_internal_links)
    report["Expected Value"].append(expected_outcomes["Broken internal links (404)"])
    report["Source"].append(sources["Broken internal links (404)"])
    report["Status"].append("❌ Fail" if broken_internal_links > 0 else "✅ Pass")

    for param in ["Broken external links", "Broken backlinks", "Broken Images"]:
        report["Category"].append("Site Health & Structure")
        report["Parameters"].append(param)
        report["Current Value"].append("N/A")
        report["Expected Value"].append(expected_outcomes[param])
        report["Source"].append(sources[param])
        report["Status"].append("ℹ️ Not Available")

    orphan_pages_count = 0
    if orphan_pages_df is not None and not orphan_pages_df.empty:
        orphan_pages_count = len(orphan_pages_df)
    report["Category"].append("Site Health & Structure")
    report["Parameters"].append("Orphan page")
    report["Current Value"].append(orphan_pages_count)
    report["Expected Value"].append(expected_outcomes["Orphan page"])
    report["Source"].append(sources["Orphan page"])
    report["Status"].append("❌ Fail" if orphan_pages_count > 0 else "✅ Pass")

    canonical_errors = len(df[df["Canonical Link Element 1"] != df["Address"]])
    report["Category"].append("Site Health & Structure")
    report["Parameters"].append("Canonical Errors")
    report["Current Value"].append(canonical_errors)
    report["Expected Value"].append(expected_outcomes["Canonical Errors"])
    report["Source"].append(sources["Canonical Errors"])
    report["Status"].append("❌ Fail" if canonical_errors > 0 else "✅ Pass")

    for param in ["Information architecture", "Header tags structure"]:
        report["Category"].append("Site Health & Structure")
        report["Parameters"].append(param)
        report["Current Value"].append("N/A")
        report["Expected Value"].append(expected_outcomes[param])
        report["Source"].append(sources[param])
        report["Status"].append("ℹ️ Not Available")

    # Link Profile & Authority
    link_profile_metrics = ["Backlinks", "Domain authority", "Spam Score"]
    for metric in link_profile_metrics:
        report["Category"].append("Link Profile & Authority")
        report["Parameters"].append(metric)
        report["Current Value"].append("N/A")
        report["Expected Value"].append(expected_outcomes[metric])
        report["Source"].append(sources[metric])
        report["Status"].append("ℹ️ Not Available")

    # Metadata & Schema analysis
    duplicate_content = 0
    if "Word Count" in df.columns and "Sentence Count" in df.columns:
        df_content = df[df["Word Count"].notna() & df["Sentence Count"].notna()]
        duplicate_content = len(
            df_content[
                df_content.duplicated(
                    subset=["Word Count", "Sentence Count"], keep=False
                )
            ]
        )
    report["Category"].append("Metadata & Schema")
    report["Parameters"].append("Duplicate content")
    report["Current Value"].append(duplicate_content)
    report["Expected Value"].append(expected_outcomes["Duplicate content"])
    report["Source"].append(sources["Duplicate content"])
    report["Status"].append("❌ Fail" if duplicate_content > 0 else "✅ Pass")

    # Alt tag analysis
    images_missing_alt_text = 0
    if alt_tag_df is not None and not alt_tag_df.empty:
        images_missing_alt_text = len(alt_tag_df)
    report["Category"].append("Metadata & Schema")
    report["Parameters"].append("Img alt tag")
    report["Current Value"].append(images_missing_alt_text)
    report["Expected Value"].append(expected_outcomes["Img alt tag"])
    report["Source"].append(sources["Img alt tag"])
    report["Status"].append("❌ Fail" if images_missing_alt_text > 0 else "✅ Pass")

    # H1 analysis
    missing_h1 = df["H1-1"].isna().sum()
    duplicate_h1 = len(df[df["H1-1"].duplicated(keep=False)])
    report["Category"].append("Metadata & Schema")
    report["Parameters"].append("Duplicate & missing H1")
    report["Current Value"].append(f"Missing: {missing_h1}, Duplicate: {duplicate_h1}")
    report["Expected Value"].append(expected_outcomes["Duplicate & missing H1"])
    report["Source"].append(sources["Duplicate & missing H1"])
    report["Status"].append(
        "❌ Fail" if missing_h1 > 0 or duplicate_h1 > 0 else "✅ Pass"
    )

    # Meta title analysis
    missing_title = df["Title 1"].isna().sum()
    duplicate_titles = 0
    df_with_titles = df[df["Title 1"].notna() & (df["Title 1"] != "")]
    duplicate_titles = len(
        df_with_titles[df_with_titles["Title 1"].duplicated(keep=False)]
    )
    report["Category"].append("Metadata & Schema")
    report["Parameters"].append("Duplicate & missing meta title")
    report["Current Value"].append(
        f"Missing: {missing_title}, Duplicate: {duplicate_titles}"
    )
    report["Expected Value"].append(expected_outcomes["Duplicate & missing meta title"])
    report["Source"].append(sources["Duplicate & missing meta title"])
    report["Status"].append(
        "❌ Fail" if missing_title > 0 or duplicate_titles > 0 else "✅ Pass"
    )

    # Meta description analysis
    missing_description = df["Meta Description 1"].isna().sum()
    duplicate_descriptions = len(df[df["Meta Description 1"].duplicated(keep=False)])
    report["Category"].append("Metadata & Schema")
    report["Parameters"].append("Duplicate & missing description")
    report["Current Value"].append(
        f"Missing: {missing_description}, Duplicate: {duplicate_descriptions}"
    )
    report["Expected Value"].append(
        expected_outcomes["Duplicate & missing description"]
    )
    report["Source"].append(sources["Duplicate & missing description"])
    report["Status"].append(
        "❌ Fail"
        if missing_description > 0 or duplicate_descriptions > 0
        else "✅ Pass"
    )

    # Schema markup analysis
    update_schema_markup_analysis(df, report, expected_outcomes, sources)

    # Prepare detailed data for deeper analysis
    duplicate_data = None
    if duplicate_titles > 0:
        duplicate_data = df_with_titles[
            df_with_titles["Title 1"].duplicated(keep=False)
        ][["Address", "Title 1", "Title 1 Length"]]

    duplicate_content_data = None
    if (
        duplicate_content > 0
        and "Word Count" in df.columns
        and "Sentence Count" in df.columns
    ):
        duplicate_content_data = df[
            df.duplicated(subset=["Word Count", "Sentence Count"], keep=False)
        ][["Address", "Word Count", "Sentence Count"]]

    h1_issues_data = None
    if missing_h1 > 0 or duplicate_h1 > 0:
        h1_issues_data = df[df["H1-1"].isna() | df["H1-1"].duplicated(keep=False)][
            ["Address", "H1-1"]
        ]
        h1_issues_data = h1_issues_data[h1_issues_data["Address"].apply(is_valid_page_url)]

    description_issues_data = None
    if missing_description > 0 or duplicate_descriptions > 0:
        description_issues_data = df[
            df["Meta Description 1"].isna()
            | df["Meta Description 1"].duplicated(keep=False)
        ][["Address", "Meta Description 1"]]
        description_issues_data = description_issues_data[description_issues_data["Address"].apply(is_valid_page_url)]

    # Convert report to list of dictionaries for JSON response
    report_list = []
    for i in range(len(report["Category"])):
        report_list.append({
            "Category": report["Category"][i],
            "Parameters": report["Parameters"][i],
            "Current Value": report["Current Value"][i],
            "Expected Value": report["Expected Value"][i],
            "Source": report["Source"][i],
            "Status": report["Status"][i]
        })

    # Convert detailed data to JSON serializable format
    detailed_data_json = {}
    if duplicate_data is not None and not duplicate_data.empty:
        detailed_data_json["duplicate_titles"] = duplicate_data.to_dict('records')
    if duplicate_content_data is not None and not duplicate_content_data.empty:
        detailed_data_json["duplicate_content"] = duplicate_content_data.to_dict('records')
    if h1_issues_data is not None and not h1_issues_data.empty:
        detailed_data_json["h1_issues"] = h1_issues_data.to_dict('records')
    if description_issues_data is not None and not description_issues_data.empty:
        detailed_data_json["description_issues"] = description_issues_data.to_dict('records')

    return report_list, detailed_data_json

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Web Audit Data Analyzer API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/validate-default-files")
async def validate_default_files() -> FileValidationResponse:
    """Check if default files exist at their expected paths"""
    missing_files = []
    
    if not os.path.exists(DATA_FILE_PATH):
        missing_files.append(f"Main data file: {DATA_FILE_PATH}")
    if not os.path.exists(ALT_TAG_DATA_PATH):
        missing_files.append(f"Alt tag data file: {ALT_TAG_DATA_PATH}")
    if not os.path.exists(ORPHAN_PAGES_DATA_PATH):
        missing_files.append(f"Orphan pages data file: {ORPHAN_PAGES_DATA_PATH}")
    
    return FileValidationResponse(
        files_exist=len(missing_files) == 0,
        missing_files=missing_files
    )

@app.post("/upload-file")
async def upload_single_file(
    file: UploadFile = File(...),
    file_type: str = Form(...)
) -> FileUploadResponse:
    """Upload and validate a single CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        return FileUploadResponse(
            message=f"{file_type} file uploaded successfully",
            rows=len(df),
            columns=list(df.columns)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/analyze-uploaded-data")
async def analyze_uploaded_data(
    main_file: UploadFile = File(...),
    alt_tag_file: UploadFile = File(...),
    orphan_file: UploadFile = File(...)
) -> AnalysisResponse:

    try:
        main_contents = await main_file.read()
        df = pd.read_csv(io.StringIO(main_contents.decode('utf-8')))
        
        alt_contents = await alt_tag_file.read()
        alt_tag_df = pd.read_csv(io.StringIO(alt_contents.decode('utf-8')))
        
        orphan_contents = await orphan_file.read()
        orphan_pages_df = pd.read_csv(io.StringIO(orphan_contents.decode('utf-8')))
        
        domain = get_domain_from_df(df)
        if not domain:
            raise HTTPException(status_code=400, detail="Cannot extract domain from main file")

        report_list, detailed_data = await asyncio.get_event_loop().run_in_executor(
            None, analyze_screaming_frog_data, df, alt_tag_df, orphan_pages_df
        )
        
        return AnalysisResponse(
            domain=domain,
            report=report_list,
            detailed_data=detailed_data,
            alt_tag_data=alt_tag_df.to_dict('records') if not alt_tag_df.empty else None,
            orphan_pages_data=orphan_pages_df.to_dict('records') if not orphan_pages_df.empty else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-default-data")
async def analyze_default_data() -> AnalysisResponse:

    try:
        if not all([
            os.path.exists(DATA_FILE_PATH),
            os.path.exists(ALT_TAG_DATA_PATH),
            os.path.exists(ORPHAN_PAGES_DATA_PATH)
        ]):
            raise HTTPException(status_code=404, detail="Default files not found")
        
        df = pd.read_csv(DATA_FILE_PATH)
        alt_tag_df = pd.read_csv(ALT_TAG_DATA_PATH)
        orphan_pages_df = pd.read_csv(ORPHAN_PAGES_DATA_PATH)
        
        domain = get_domain_from_df(df)
        if not domain:
            raise HTTPException(status_code=400, detail="Cannot extract domain from data")
        
        report_list, detailed_data = await asyncio.get_event_loop().run_in_executor(
            None, analyze_screaming_frog_data, df, alt_tag_df, orphan_pages_df
        )
        
        return AnalysisResponse(
            domain=domain,
            report=report_list,
            detailed_data=detailed_data,
            alt_tag_data=alt_tag_df.to_dict('records') if not alt_tag_df.empty else None,
            orphan_pages_data=orphan_pages_df.to_dict('records') if not orphan_pages_df.empty else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
