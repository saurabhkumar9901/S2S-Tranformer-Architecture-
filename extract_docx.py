import xml.etree.ElementTree as ET
import sys
import io

def extract_text(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Modern .docx uses these namespaces
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        text = []
        for paragraph in root.findall('.//w:p', ns):
            p_text = []
            for run in paragraph.findall('.//w:t', ns):
                if run.text:
                    p_text.append(run.text)
            text.append("".join(p_text))
        
        return "\n".join(text)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    # Force UTF-8 encoding for stdout
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    if len(sys.argv) < 2:
        print("Usage: python extract_docx.py <path_to_xml>")
        sys.exit(1)
    
    print(extract_text(sys.argv[1]))
