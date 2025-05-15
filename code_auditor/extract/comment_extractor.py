# extract/comment_extractor.py

import os
import json
from typing import List, Dict, Any
import re
from pathlib import Path

class CommentExtractor:
    """Extract comments from source code files using regular expressions and (optionally) tree-sitter."""
    
    def __init__(self):
        """Initialize the comment extractor with language configurations."""
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cs': 'c_sharp',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.html': 'html',
            '.css': 'css',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
        }
        
        # Try to initialize tree-sitter if available
        self.use_tree_sitter = False
        try:
            import tree_sitter
            from tree_sitter import Language, Parser
            
            # In a real implementation, you'd build language libraries
            # This is just a placeholder for the structure
            self.parsers = {}
            self.use_tree_sitter = True
            print("Tree-sitter available. Using advanced parsing.")
        except ImportError:
            print("Tree-sitter not available. Falling back to regex parsing.")
    
    def extract_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract comments from a single file."""
        _, ext = os.path.splitext(file_path)
        
        if ext not in self.supported_extensions:
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
            
        # Use regex patterns for comment extraction
        comments = []
        
        # Python comments
        if ext == '.py':
            # Single line comments
            single_line = re.finditer(r'#(.*?)$', content, re.MULTILINE)
            for match in single_line:
                comments.append({
                    'text': match.group(1).strip(),
                    'type': 'single_line',
                    'file': file_path,
                    'line': content[:match.start()].count('\n') + 1
                })
                
            # Docstrings
            docstrings = re.finditer(r'"""(.*?)"""', content, re.DOTALL)
            for match in docstrings:
                comments.append({
                    'text': match.group(1).strip(),
                    'type': 'docstring',
                    'file': file_path,
                    'line': content[:match.start()].count('\n') + 1
                })
                
            # Single quotes docstrings
            docstrings_single = re.finditer(r"'''(.*?)'''", content, re.DOTALL)
            for match in docstrings_single:
                comments.append({
                    'text': match.group(1).strip(),
                    'type': 'docstring',
                    'file': file_path,
                    'line': content[:match.start()].count('\n') + 1
                })
                
        # C-style comments (JavaScript, Java, C#, C++, C)
        elif ext in ['.js', '.ts', '.java', '.cs', '.cpp', '.c', '.php', '.kt', '.swift']:
            # Single line comments
            single_line = re.finditer(r'//(.*?)$', content, re.MULTILINE)
            for match in single_line:
                comments.append({
                    'text': match.group(1).strip(),
                    'type': 'single_line',
                    'file': file_path,
                    'line': content[:match.start()].count('\n') + 1
                })
                
            # Multi-line comments
            multi_line = re.finditer(r'/\*(.*?)\*/', content, re.DOTALL)
            for match in multi_line:
                comments.append({
                    'text': match.group(1).strip(),
                    'type': 'multi_line',
                    'file': file_path,
                    'line': content[:match.start()].count('\n') + 1
                })
                
        # Ruby comments
        elif ext == '.rb':
            # Single line comments
            single_line = re.finditer(r'#(.*?)$', content, re.MULTILINE)
            for match in single_line:
                comments.append({
                    'text': match.group(1).strip(),
                    'type': 'single_line',
                    'file': file_path,
                    'line': content[:match.start()].count('\n') + 1
                })
                
            # Multi-line comments
            multi_line = re.finditer(r'=begin(.*?)=end', content, re.DOTALL)
            for match in multi_line:
                comments.append({
                    'text': match.group(1).strip(),
                    'type': 'multi_line',
                    'file': file_path,
                    'line': content[:match.start()].count('\n') + 1
                })
                
        # HTML comments
        elif ext == '.html':
            # HTML comments
            html_comments = re.finditer(r'<!--(.*?)-->', content, re.DOTALL)
            for match in html_comments:
                comments.append({
                    'text': match.group(1).strip(),
                    'type': 'html_comment',
                    'file': file_path,
                    'line': content[:match.start()].count('\n') + 1
                })
                
        # CSS comments
        elif ext == '.css':
            # CSS comments
            css_comments = re.finditer(r'/\*(.*?)\*/', content, re.DOTALL)
            for match in css_comments:
                comments.append({
                    'text': match.group(1).strip(),
                    'type': 'css_comment',
                    'file': file_path,
                    'line': content[:match.start()].count('\n') + 1
                })
                
        return comments
    
    def extract_from_directory(self, directory: str, output_file: str = None, ignore_dirs: List[str] = None) -> List[Dict[str, Any]]:
        """
        Extract comments from all files in a directory recursively.
        
        Args:
            directory: Path to the directory to scan
            output_file: Optional path to save the results as JSON
            ignore_dirs: List of directory names to ignore (e.g., ['node_modules', '.git'])
            
        Returns:
            List of dictionaries containing comment data
        """
        if ignore_dirs is None:
            ignore_dirs = ['node_modules', '.git', 'venv', '__pycache__', 'build', 'dist']
            
        all_comments = []
        
        for root, dirs, files in os.walk(directory):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_comments = self.extract_from_file(file_path)
                    all_comments.extend(file_comments)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_comments, f, indent=2)
                
        return all_comments
    
    def analyze_comment_patterns(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in extracted comments.
        
        Args:
            comments: List of comment dictionaries from extract_from_directory
            
        Returns:
            Dictionary with analysis results
        """
        if not comments:
            return {'total': 0}
            
        analysis = {
            'total': len(comments),
            'by_file_extension': {},
            'by_type': {},
            'potential_todos': [],
            'potential_fixmes': [],
            'potential_technical_debt': []
        }
        
        # Analyze by file extension and type
        for comment in comments:
            file = comment.get('file', '')
            _, ext = os.path.splitext(file)
            
            if ext not in analysis['by_file_extension']:
                analysis['by_file_extension'][ext] = 0
            analysis['by_file_extension'][ext] += 1
            
            comment_type = comment.get('type', 'unknown')
            if comment_type not in analysis['by_type']:
                analysis['by_type'][comment_type] = 0
            analysis['by_type'][comment_type] += 1
            
            # Check for TODOs, FIXMEs and technical debt indicators
            text = comment.get('text', '').lower()
            
            if 'todo' in text:
                analysis['potential_todos'].append({
                    'file': file,
                    'line': comment.get('line', 0),
                    'text': comment.get('text', '')
                })
                
            if any(marker in text for marker in ['fixme', 'fix me', 'needs fix']):
                analysis['potential_fixmes'].append({
                    'file': file,
                    'line': comment.get('line', 0),
                    'text': comment.get('text', '')
                })
                
            if any(phrase in text for phrase in [
                'technical debt', 'hack', 'workaround', 'temporary', 'band-aid',
                'quick fix', 'needs refactor', 'refactor later', 'not optimal'
            ]):
                analysis['potential_technical_debt'].append({
                    'file': file,
                    'line': comment.get('line', 0),
                    'text': comment.get('text', '')
                })
                
        return analysis

# Example usage
if __name__ == "__main__":
    extractor = CommentExtractor()
    comments = extractor.extract_from_directory("./")
    print(f"Extracted {len(comments)} comments")
    
    # Save to file
    with open("comments.json", "w", encoding="utf-8") as f:
        json.dump(comments, f, indent=2)