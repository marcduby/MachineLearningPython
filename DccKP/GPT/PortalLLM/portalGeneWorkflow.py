import requests
from bs4 import BeautifulSoup
import pdfkit
import os
from typing import Optional
import anthropic

class WebPageAnalyzer:
    def __init__(self, anthropic_api_key: Optional[str] = None):
        """
        Initialize the WebPageAnalyzer with optional Anthropic API key.
        
        :param anthropic_api_key: API key for Anthropic's Claude
        """
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required. Set it in environment or pass directly.")
        
        # Ensure pdfkit is configured (you may need to install wkhtmltopdf)
        self.pdfkit_config = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')

    def download_web_page(self, url: str) -> str:
        """
        Download the content of a web page.
        
        :param url: Web page URL to download
        :return: HTML content of the page
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error downloading web page: {e}")
            return ""

    def convert_to_pdf(self, html_content: str, output_path: str = 'webpage.pdf') -> str:
        """
        Convert HTML content to PDF.
        
        :param html_content: HTML content to convert
        :param output_path: Path to save the PDF
        :return: Path to the generated PDF
        """
        try:
            pdfkit.from_string(html_content, output_path, configuration=self.pdfkit_config)
            return output_path
        except Exception as e:
            print(f"Error converting to PDF: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyPDF2.
        
        :param pdf_path: Path to the PDF file
        :return: Extracted text from PDF
        """
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except ImportError:
            print("PyPDF2 is not installed. Please install it to extract PDF text.")
            return ""
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def analyze_with_claude(self, text: str, analysis_prompt: str = "Summarize the key points of this text.") -> str:
        """
        Analyze text using Anthropic's Claude LLM.
        
        :param text: Text to analyze
        :param analysis_prompt: Prompt for the LLM analysis
        :return: LLM's analysis result
        """
        try:
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user", 
                        "content": f"{analysis_prompt}\n\nText to analyze:\n{text}"
                    }
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            print(f"Error analyzing with Claude: {e}")
            return ""

    def process_webpage(self, url: str, analysis_prompt: Optional[str] = None) -> str:
        """
        Full workflow: download webpage, convert to PDF, extract text, and analyze.
        
        :param url: Web page URL to process
        :param analysis_prompt: Optional custom analysis prompt
        :return: LLM analysis result
        """
        # Download web page
        html_content = self.download_web_page(url)
        if not html_content:
            return "Failed to download web page."

        # Convert to PDF
        pdf_path = self.convert_to_pdf(html_content)
        if not pdf_path:
            return "Failed to convert to PDF."

        # Extract text from PDF
        extracted_text = self.extract_text_from_pdf(pdf_path)
        if not extracted_text:
            return "Failed to extract text from PDF."

        # Analyze with Claude
        default_prompt = "Provide a comprehensive summary of the key points in this text."
        analysis_result = self.analyze_with_claude(
            extracted_text, 
            analysis_prompt or default_prompt
        )

        # Optional: Clean up PDF file
        os.remove(pdf_path)

        return analysis_result

# Example usage
def main():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    analyzer = WebPageAnalyzer()  # Assumes ANTHROPIC_API_KEY is set in environment
    
    result = analyzer.process_webpage(
        url, 
        "Extract the top 5 most important historical developments in AI from this text."
    )
    print(result)

if __name__ == "__main__":
    main()


    