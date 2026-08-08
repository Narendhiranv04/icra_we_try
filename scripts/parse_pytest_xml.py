import xml.etree.ElementTree as ET
import json

def main():
    try:
        tree = ET.parse("artifacts/learning_stage1/pytest.xml")
        root = tree.getroot()
        
        # usually <testsuite> is the root, or <testsuites> contains <testsuite>
        testsuite = root if root.tag == 'testsuite' else root.find('.//testsuite')
        
        summary = {
            "total": int(testsuite.get("tests", 0)),
            "passed": int(testsuite.get("tests", 0)) - int(testsuite.get("failures", 0)) - int(testsuite.get("errors", 0)) - int(testsuite.get("skipped", 0)),
            "failed": int(testsuite.get("failures", 0)) + int(testsuite.get("errors", 0)),
            "skipped": int(testsuite.get("skipped", 0)),
            "runtime": float(testsuite.get("time", 0.0))
        }
        
        with open("artifacts/learning_stage1/test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"Test summary saved: {summary}")
    except Exception as e:
        print(f"Failed to parse pytest xml: {e}")

if __name__ == "__main__":
    main()
