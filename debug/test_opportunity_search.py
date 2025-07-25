#!/usr/bin/env python3
"""
Test script for opportunity search functionality
"""

import json
import subprocess
import sys

def test_opportunity_search():
    """Test the opportunity search endpoint with various configurations"""
    
    base_url = "http://192.168.15.206:5000/api/v1/search/opportunity-search"
    
    test_cases = [
        {
            "name": "Original Request (should work now)",
            "data": {
                "opportunity_ids": ["0B62CE5D-C3FD-44D0-ADDC-015195EC28BD"],
                "title_similarity_threshold": 0.35,
                "description_similarity_threshold": 0.4,
                "document_similarity_threshold": 0.4,
                "start_posted_date": "2020-01-01", 
                "end_posted_date": "2025-12-31",
                "document_sow_boost_multiplier": 2,
                "limit": 10
            }
        },
        {
            "name": "Minimal Request",
            "data": {
                "opportunity_ids": ["0B62CE5D-C3FD-44D0-ADDC-015195EC28BD"]
            }
        },
        {
            "name": "Multiple Opportunities",
            "data": {
                "opportunity_ids": [
                    "0B62CE5D-C3FD-44D0-ADDC-015195EC28BD",
                    "F19A27DC-E75B-4F76-B268-00C16CCFF02B"
                ],
                "limit": 5
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n=== {test_case['name']} ===")
        
        # Prepare curl command
        json_data = json.dumps(test_case['data'])
        
        cmd = [
            'curl', '-s', '-X', 'POST',
            base_url,
            '-H', 'accept: application/json',
            '-H', 'Content-Type: application/json',
            '-d', json_data
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                try:
                    response = json.loads(result.stdout)
                    
                    if 'error' in response:
                        print(f"❌ Error: {response['error']}")
                    elif response.get('results') is None:
                        print(f"❌ Null response: {response}")
                    else:
                        results = response.get('results', [])
                        print(f"✅ Success: Found {len(results)} results")
                        print(f"   Processing time: {response.get('processing_time_ms', 'N/A')}ms")
                        print(f"   Request ID: {response.get('request_id', 'N/A')}")
                        
                        # Show top 3 results
                        for i, result in enumerate(results[:3]):
                            combined_score = result.get('combined_score', 0)
                            doc_score = result.get('document_score', 0)
                            opp_id = result.get('opportunity_id', 'N/A')
                            doc_count = result.get('document_match_count', 0)
                            print(f"   {i+1}. {opp_id}: Combined={combined_score:.3f}, Doc={doc_score:.3f}, Matches={doc_count}")
                            
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON response: {result.stdout[:200]}...")
            else:
                print(f"❌ Curl failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"❌ Request timed out")
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print(f"\n=== Test Summary ===")
    print("If the 'Original Request' now shows ✅ Success, the validation fix worked!")

if __name__ == "__main__":
    test_opportunity_search()
