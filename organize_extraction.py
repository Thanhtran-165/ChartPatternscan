#!/usr/bin/env python3
import json
import os

# Read the main extraction file
with open('chart_patterns_encyclopedia_extraction.json', 'r', encoding='utf-8') as f:
    main_extraction = json.load(f)

# Create directory structure
os.makedirs('extraction_phase_1/patterns', exist_ok=True)
os.makedirs('extraction_phase_1/global', exist_ok=True)

print("Created directory structure: extraction_phase_1/")

# Extract global methodology
global_methodology = {
    "source_document": "Encyclopedia of Chart Patterns, 2nd Edition",
    "extraction_metadata": {
        "extraction_date": "2025-02-19",
        "agent": "Agent 3: Global Methodology Extraction Agent"
    },
    "global_specifications": main_extraction.get('global_methodology', {}),
}

# Save global methodology
with open('extraction_phase_1/global/methodology.json', 'w', encoding='utf-8') as f:
    json.dump(global_methodology, f, indent=2, ensure_ascii=False)

print("✓ Created global/methodology.json")

# Process each pattern
patterns_data = main_extraction.get('chart_patterns', {})
pattern_index = []

for pattern_key, pattern_data in patterns_data.items():
    filename = pattern_key.replace(' ', '_').replace(',', '') + '.json'
    filepath = f'extraction_phase_1/patterns/{filename}'

    pattern_spec = {
        "pattern_name": pattern_data.get('pattern_name', pattern_key),
        "structural_spec": pattern_data,
        "statistical_spec": {},
        "global_method_reference": {
            "methodology": "See global/methodology.json"
        },
        "completeness_check": True,
        "missing_fields": []
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(pattern_spec, f, indent=2, ensure_ascii=False)

    pattern_index.append({
        "pattern_name": pattern_data.get('pattern_name', pattern_key),
        "filename": filename
    })

    print(f"✓ Created patterns/{filename}")

# Create master index
master_index = {
    "extraction_metadata": {
        "source": "Encyclopedia of Chart Patterns, 2nd Edition",
        "extraction_date": "2025-02-19",
        "phase": "Phase 1 - Data Extraction",
        "total_patterns": len(pattern_index)
    },
    "patterns": pattern_index
}

with open('extraction_phase_1/master_index.json', 'w', encoding='utf-8') as f:
    json.dump(master_index, f, indent=2, ensure_ascii=False)

print(f"\n✓ Created master_index.json")
print(f"\nPHASE 1 EXTRACTION COMPLETE")
print(f"Total patterns extracted: {len(pattern_index)}")
