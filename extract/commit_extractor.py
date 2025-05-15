# extract/commit_extractor.py
import git
import json
from typing import List, Dict, Any
from datetime import datetime
import re

class CommitExtractor:
    """Extract commit messages and metadata from git repositories."""
    
    def __init__(self, repo_path: str):
        """Initialize with the path to a git repository."""
        try:
            self.repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            raise ValueError(f"{repo_path} is not a valid git repository")
    
    def extract_commits(self, max_count: int = None, branch: str = 'main', output_file: str = None) -> List[Dict[str, Any]]:
        """
        Extract commit messages and metadata.
        
        Args:
            max_count: Maximum number of commits to extract (newest first)
            branch: Branch name to extract from
            output_file: Optional path to save results
            
        Returns:
            List of dictionaries containing commit data
        """
        try:
            commits = []
            for commit in self.repo.iter_commits(branch, max_count=max_count):
                # Extract basic commit info
                commit_data = {
                    'hash': commit.hexsha,
                    'short_hash': commit.hexsha[:7],
                    'author': commit.author.name,
                    'author_email': commit.author.email,
                    'date': commit.committed_datetime.isoformat(),
                    'message': commit.message,
                    'stats': {
                        'files_changed': len(commit.stats.files),
                        'insertions': commit.stats.total['insertions'],
                        'deletions': commit.stats.total['deletions'],
                    }
                }
                
                # Extract structured commit message if following conventional commits
                conventional_match = re.match(
                    r'^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(?:\((?P<scope>[^)]+)\))?: (?P<description>.+)',
                    commit.message.splitlines()[0]
                )
                
                if conventional_match:
                    commit_data['conventional'] = {
                        'type': conventional_match.group('type'),
                        'scope': conventional_match.group('scope'),
                        'description': conventional_match.group('description')
                    }
                
                commits.append(commit_data)
            
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(commits, f, indent=2)
                    
            return commits
        except Exception as e:
            print(f"Error extracting commits: {e}")
            return []
    
    def extract_author_timeline(self, author_email: str = None, days: int = 30) -> Dict[str, Any]:
        """Extract commit activity timeline for a specific author."""
        if not author_email:
            return {}
            
        since_date = datetime.now() - datetime.timedelta(days=days)
        
        commits = []
        for commit in self.repo.iter_commits(author=author_email, since=since_date):
            commits.append({
                'hash': commit.hexsha[:7],
                'date': commit.committed_datetime.isoformat(),
                'message': commit.message.splitlines()[0],
                'stats': {
                    'files_changed': len(commit.stats.files),
                    'insertions': commit.stats.total['insertions'],
                    'deletions': commit.stats.total['deletions'],
                }
            })
            
        # Organize by date
        commits_by_date = {}
        for commit in commits:
            date = commit['date'].split('T')[0]  # Just the date part
            if date not in commits_by_date:
                commits_by_date[date] = []
            commits_by_date[date].append(commit)
            
        return {
            'author_email': author_email,
            'total_commits': len(commits),
            'timeline': commits_by_date
        }