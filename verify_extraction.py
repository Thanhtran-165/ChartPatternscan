#!/usr/bin/env python3
"""
Verify Phase 1 extraction completeness.
"""

import json
import os

def verify_pattern_file(filepath, filename):
    """Verify a single pattern file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    issues = []

    # Check structural_spec fields
    if 'structural_spec' not in data:
        issues.append("Missing structural_spec")
    else:
        required_structural = [
            'pattern_name', 'prior_trend_requirement', 'geometry_rules',
            'touch_requirements', 'sequence_logic', 'trendline_constraints',
            'volume_behavior', 'breakout_condition', 'duration_constraints',
            'height_definition', 'width_definition', 'variants',
            'invalidation_conditions', 'partial_move_definitions',
            'intraformation_failure_rules'
        ]
        for field in required_structural:
            if field not in data['structural_spec']:
                issues.append(f"Missing structural field: {field}")
            elif not data['structural_spec'][field]:
                issues.append(f"Empty structural field: {field}")

    # Check statistical_spec fields
    if 'statistical_spec' not in data:
        issues.append("Missing statistical_spec")
    else:
        required_statistical = [
            'average_rise_method', 'average_decline_method', 'failure_definition',
            'failure_thresholds', 'ultimate_high_definition', 'ultimate_low_definition',
            'time_to_high_method', 'time_to_low_method', 'throwback_definition',
            'pullback_definition', 'gap_impact_method', 'height_effect_rule',
            'width_effect_rule', 'busted_pattern_definition', 'post_trend_measurement',
            'performance_ranking_method', 'regime_separation_rule',
            'frequency_distribution_method', 'sample_filtering_rule', 'bias_controls'
        ]
        for field in required_statistical:
            if field not in data['statistical_spec']:
                issues.append(f"Missing statistical field: {field}")
            elif not data['statistical_spec'][field] and not isinstance(data['statistical_spec'][field], list):
                issues.append(f"Empty statistical field: {field}")

    # Check metadata
    if 'missing_fields' not in data:
        issues.append("Missing missing_fields metadata")
    elif data['missing_fields'] != []:
        issues.append(f"missing_fields is not empty: {data['missing_fields']}")

    if 'completeness_check' not in data:
        issues.append("Missing completeness_check metadata")
    elif data['completeness_check'] != True:
        issues.append("completeness_check is not True")

    return len(issues) == 0, issues

def verify_global_methodology(filepath):
    """Verify global methodology file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    issues = []
    required_fields = [
        'breakout_logic', 'overlap_handling', 'market_regime_definition',
        'data_filtering', 'price_adjustment', 'lookahead_control',
        'averaging_method', 'failure_classification', 'sample_selection',
        'time_measurement_definition', 'trend_change_definition',
        'extreme_price_definition', 'outlier_handling', 'ranking_logic'
    ]

    for field in required_fields:
        if field not in data.get('global_specifications', {}):
            issues.append(f"Missing global field: {field}")
        elif not data['global_specifications'][field]:
            issues.append(f"Empty global field: {field}")

    return len(issues) == 0, issues

def main():
    base_dir = "/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet/Nghiên cứu mô hình nến"

    print("=" * 60)
    print("PHASE 1 EXTRACTION VERIFICATION")
    print("=" * 60)
    print()

    # Verify global methodology
    print("1. GLOBAL METHODOLOGY")
    print("-" * 60)
    global_path = os.path.join(base_dir, "extraction_phase_1/global/methodology.json")
    success, issues = verify_global_methodology(global_path)
    if success:
        print("✅ Global methodology: COMPLETE (14/14 fields)")
    else:
        print(f"❌ Global methodology: INCOMPLETE")
        for issue in issues:
            print(f"   - {issue}")
    print()

    # Verify pattern files
    print("2. PATTERN FILES")
    print("-" * 60)
    patterns_dir = os.path.join(base_dir, "extraction_phase_1/patterns")
    pattern_files = [f for f in os.listdir(patterns_dir) if f.endswith('.json')]

    complete_patterns = []
    incomplete_patterns = []

    for filename in sorted(pattern_files):
        filepath = os.path.join(patterns_dir, filename)
        success, issues = verify_pattern_file(filepath, filename)
        if success:
            complete_patterns.append(filename)
        else:
            incomplete_patterns.append((filename, issues))

    print(f"Total pattern files: {len(pattern_files)}")
    print(f"✅ Complete: {len(complete_patterns)}")
    print(f"❌ Incomplete: {len(incomplete_patterns)}")
    print()

    if incomplete_patterns:
        print("Incomplete patterns:")
        for filename, issues in incomplete_patterns:
            print(f"  ❌ {filename}")
            for issue in issues[:5]:  # Show first 5 issues
                print(f"     - {issue}")
    print()

    # Verify master file
    print("3. MASTER EXTRACTION FILE")
    print("-" * 60)
    master_path = os.path.join(base_dir, "complete_extraction_phase1.json")
    if os.path.exists(master_path):
        with open(master_path, 'r') as f:
            master_data = json.load(f)
        patterns = master_data.get('patterns', [])
        print(f"✅ Master file exists")
        print(f"   - Patterns in master: {len(patterns)}")
        print(f"   - Expected patterns: 24")
        if len(patterns) == 24:
            print("   ✅ Pattern count matches")
        else:
            print(f"   ❌ Pattern count mismatch")

        # Check summary
        summary = master_data.get('extraction_summary', {})
        print(f"   - Total patterns extracted: {summary.get('total_patterns_extracted', 'N/A')}")
        print(f"   - Patterns with complete structural_spec: {summary.get('patterns_with_complete_structural_spec', 'N/A')}")
        print(f"   - Patterns with complete statistical_spec: {summary.get('patterns_with_complete_statistical_spec', 'N/A')}")
        print(f"   - Completeness status: {summary.get('completeness_status', 'N/A')}")
    else:
        print("❌ Master file not found")
    print()

    # Final summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print()

    all_complete = (
        success and  # Global methodology complete
        len(incomplete_patterns) == 0 and  # All patterns complete
        len(complete_patterns) == 24  # All 24 patterns present
    )

    if all_complete:
        print("🎉 PHASE 1 EXTRACTION: 100% COMPLETE")
        print()
        print("✅ Global methodology: 14/14 fields")
        print(f"✅ Pattern files: 24/24 complete")
        print(f"✅ Structural spec: 15/15 fields per pattern")
        print(f"✅ Statistical spec: 20/20 fields per pattern")
        print(f"✅ Total fields: 854/854 filled")
        print()
        print("All pattern files have been successfully updated with:")
        print("  • Complete structural_spec (15 fields)")
        print("  • Complete statistical_spec (20 fields)")
        print("  • missing_fields: []")
        print("  • completeness_check: true")
    else:
        print("⚠️  PHASE 1 EXTRACTION: INCOMPLETE")
        print()
        print("Please review the issues listed above.")

if __name__ == "__main__":
    main()
