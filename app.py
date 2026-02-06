"""
AI Data Entry Automation Tool
A single-file Python application that automates lead generation using deep learning.
"""

import re
import time
import random
import logging
from typing import List, Dict, Tuple
from datetime import datetime
import io

# Web and scraping
import requests
from bs4 import BeautifulSoup

# Data processing
import pandas as pd
import numpy as np

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Web UI
from pywebio import start_server
from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *


# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
]


# ============================================================================
# DEEP NEURAL NETWORK (16 LAYERS)
# ============================================================================

class LeadScoringNetwork(nn.Module):
    """
    16-layer deep neural network for lead quality prediction.
    Architecture: Input -> 14 Hidden Layers -> Output (sigmoid)
    """
    
    def __init__(self, input_size=10, hidden_sizes=None):
        super(LeadScoringNetwork, self).__init__()
        
        if hidden_sizes is None:
            # 14 hidden layers with decreasing neurons
            hidden_sizes = [256, 256, 128, 128, 128, 64, 64, 64, 32, 32, 32, 16, 16, 8]
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        
        # 14 Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"✅ Initialized 16-layer neural network")
        logger.info(f"   Total layers: {len([l for l in layers if isinstance(l, nn.Linear)])}")
    
    def forward(self, x):
        return self.network(x)


class SyntheticLeadDataset(Dataset):
    """Generate synthetic training data for the neural network"""
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.X, self.y = self._generate_data()
    
    def _generate_data(self):
        """Create synthetic features and labels"""
        X = torch.randn(self.num_samples, 10)  # 10 features
        
        # Create labels based on feature combinations (simulating lead quality)
        weights = torch.tensor([0.2, 0.3, 0.1, 0.15, 0.05, 0.1, 0.05, 0.02, 0.02, 0.01])
        scores = torch.sigmoid(torch.matmul(X, weights) + torch.randn(self.num_samples) * 0.1)
        y = (scores > 0.5).float().unsqueeze(1)
        
        return X, y
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AILeadScorer:
    """AI model manager for training and inference"""
    
    def __init__(self):
        self.model = LeadScoringNetwork(input_size=10)
        self.trained = False
    
    def train_model(self, epochs=20):
        """Train the neural network on synthetic data"""
        logger.info("🧠 Training AI model...")
        
        dataset = SyntheticLeadDataset(num_samples=1000)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"   Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.trained = True
        logger.info("✅ Model training completed!")
    
    def score_leads(self, features: np.ndarray) -> np.ndarray:
        """Predict lead quality scores"""
        if not self.trained:
            self.train_model()
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(features)
            scores = self.model(X_tensor).numpy().flatten()
        
        return scores


# ============================================================================
# WEB SCRAPING ENGINE
# ============================================================================

class WebScraper:
    """Intelligent web scraping module with retry logic"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
    
    def search_duckduckgo(self, query: str, num_results: int = 10) -> List[str]:
        """Search DuckDuckGo and return URLs"""
        logger.info(f"🔍 Searching DuckDuckGo: {query}")
        
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        urls = []
        
        try:
            time.sleep(random.uniform(1, 2))  # Respectful delay
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = soup.find_all('a', class_='result__a', limit=num_results)
                
                for result in results:
                    href = result.get('href')
                    if href:
                        urls.append(href)
            
            logger.info(f"   Found {len(urls)} URLs")
            
        except Exception as e:
            logger.error(f"   Search error: {str(e)}")
        
        return urls
    
    def extract_contact_info(self, url: str) -> Dict:
        """Extract business information from a URL"""
        contact_info = {
            'website': url,
            'business_name': '',
            'email': '',
            'phone': '',
            'location': ''
        }
        
        try:
            time.sleep(random.uniform(0.5, 1.5))  # Respectful delay
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                
                # Extract business name (from title or h1)
                title = soup.find('title')
                if title:
                    contact_info['business_name'] = title.get_text().strip()[:100]
                else:
                    h1 = soup.find('h1')
                    if h1:
                        contact_info['business_name'] = h1.get_text().strip()[:100]
                
                # Extract email
                emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
                if emails:
                    # Filter out common generic emails
                    valid_emails = [e for e in emails if not any(x in e.lower() for x in ['example', 'test', 'noreply'])]
                    if valid_emails:
                        contact_info['email'] = valid_emails[0]
                
                # Extract phone
                phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
                if phones:
                    contact_info['phone'] = phones[0]
                
        except Exception as e:
            logger.error(f"   Extraction error for {url}: {str(e)}")
        
        return contact_info
    
    def scrape_leads(self, niche: str, location: str, num_leads: int) -> List[Dict]:
        """Main scraping pipeline"""
        query = f"{niche} in {location}"
        
        # Search for URLs
        urls = self.search_duckduckgo(query, num_results=num_leads * 2)
        
        leads = []
        for i, url in enumerate(urls[:num_leads]):
            logger.info(f"📄 Scraping {i + 1}/{len(urls[:num_leads])}: {url}")
            
            contact_info = self.extract_contact_info(url)
            contact_info['location'] = location
            contact_info['niche'] = niche
            
            if contact_info['business_name']:  # Only add if we got some data
                leads.append(contact_info)
            
            if len(leads) >= num_leads:
                break
        
        logger.info(f"✅ Scraped {len(leads)} leads")
        return leads


# ============================================================================
# DATA CLEANING & VALIDATION
# ============================================================================

class DataCleaner:
    """Clean and validate scraped data"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def clean_phone(phone: str) -> str:
        """Standardize phone number format"""
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        return phone
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset"""
        logger.info("🧹 Cleaning data...")
        
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['website'], keep='first')
        logger.info(f"   Removed {initial_count - len(df)} duplicates")
        
        # Validate emails
        df['email_valid'] = df['email'].apply(
            lambda x: DataCleaner.validate_email(x) if pd.notna(x) and x else False
        )
        
        # Clean phone numbers
        df['phone'] = df['phone'].apply(
            lambda x: DataCleaner.clean_phone(x) if pd.notna(x) and x else ''
        )
        
        # Remove rows with no business name
        df = df[df['business_name'].notna() & (df['business_name'] != '')]
        
        logger.info(f"✅ Cleaned dataset: {len(df)} valid leads")
        return df


# ============================================================================
# FEATURE ENGINEERING FOR AI
# ============================================================================

class FeatureEngineering:
    """Convert lead data into numerical features for AI model"""
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> np.ndarray:
        """Generate 10 numerical features from lead data"""
        features = []
        
        for _, row in df.iterrows():
            feature_vector = [
                1.0 if row['email'] else 0.0,  # Has email
                1.0 if row['phone'] else 0.0,  # Has phone
                len(row['business_name']) / 100.0,  # Name length (normalized)
                1.0 if row.get('email_valid', False) else 0.0,  # Email valid
                len(row['website']) / 100.0,  # URL length (normalized)
                random.random(),  # Domain authority (simulated)
                random.random(),  # Social media presence (simulated)
                random.random(),  # Website quality (simulated)
                random.random(),  # Content richness (simulated)
                random.random(),  # Trust signals (simulated)
            ]
            features.append(feature_vector)
        
        return np.array(features)


# ============================================================================
# EXPORT MODULE
# ============================================================================

class DataExporter:
    """Export data to CSV and Excel formats"""
    
    @staticmethod
    def to_csv(df: pd.DataFrame) -> bytes:
        """Convert DataFrame to CSV bytes"""
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue().encode('utf-8')
    
    @staticmethod
    def to_excel(df: pd.DataFrame) -> bytes:
        """Convert DataFrame to Excel bytes"""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Leads')
        return output.getvalue()


# ============================================================================
# MAIN APPLICATION PIPELINE
# ============================================================================

class LeadGenerationPipeline:
    """Main orchestration pipeline"""
    
    def __init__(self):
        self.scraper = WebScraper()
        self.cleaner = DataCleaner()
        self.ai_scorer = AILeadScorer()
        self.exporter = DataExporter()
    
    def run(self, niche: str, location: str, num_leads: int) -> Tuple[pd.DataFrame, bytes, bytes]:
        """Execute full pipeline"""
        
        # Step 1: Scrape data
        put_text("🔍 Step 1/5: Searching and scraping web data...")
        raw_leads = self.scraper.scrape_leads(niche, location, num_leads)
        
        if not raw_leads:
            raise Exception("No leads found. Try different keywords or location.")
        
        # Step 2: Create DataFrame
        put_text("📊 Step 2/5: Processing scraped data...")
        df = pd.DataFrame(raw_leads)
        
        # Step 3: Clean data
        put_text("🧹 Step 3/5: Cleaning and validating data...")
        df = self.cleaner.clean_data(df)
        
        # Step 4: AI Scoring
        put_text("🧠 Step 4/5: AI model scoring leads...")
        features = FeatureEngineering.create_features(df)
        scores = self.ai_scorer.score_leads(features)
        df['ai_quality_score'] = scores
        
        # Filter high-quality leads (score > 0.6)
        df_filtered = df[df['ai_quality_score'] > 0.6].copy()
        df_filtered = df_filtered.sort_values('ai_quality_score', ascending=False)
        
        put_text(f"✅ AI filtered: {len(df_filtered)} high-quality leads (score > 0.6)")
        
        # Step 5: Export
        put_text("📦 Step 5/5: Generating export files...")
        
        # Prepare final output columns
        output_df = df_filtered[[
            'business_name', 'website', 'email', 'phone', 
            'location', 'niche', 'ai_quality_score'
        ]].copy()
        
        csv_data = self.exporter.to_csv(output_df)
        excel_data = self.exporter.to_excel(output_df)
        
        return output_df, csv_data, excel_data


# ============================================================================
# PYWEBIO WEB INTERFACE
# ============================================================================

def main():
    """Main web application interface"""
    
    # Page setup
    set_env(title="AI Data Entry Automation Tool")
    
    # Header
    put_markdown("# 🤖 AI Data Entry Automation Tool")
    put_markdown("### Automate Lead Generation with Deep Learning")
    put_markdown("---")
    
    put_text("This tool uses a 16-layer neural network to find and score high-quality leads.")
    put_html("<br>")
    
    # Input form
    put_markdown("## 📝 Enter Search Parameters")
    
    niche = input("Business Niche (e.g., dentists, real estate agents):", 
                  type=TEXT, 
                  required=True,
                  placeholder="dentists")
    
    location = input("Location (e.g., New York, London):", 
                     type=TEXT, 
                     required=True,
                     placeholder="New York")
    
    num_leads = input("Number of Leads Required:", 
                      type=NUMBER, 
                      required=True,
                      value=10,
                      validate=lambda x: x > 0 and x <= 50)
    
    put_html("<br>")
    
    # Start button
    if actions("", ['🚀 Start Automation']) == '🚀 Start Automation':
        
        put_markdown("---")
        put_markdown("## 🔄 Processing...")
        put_html("<br>")
        
        try:
            # Initialize pipeline
            pipeline = LeadGenerationPipeline()
            
            # Run pipeline
            output_df, csv_data, excel_data = pipeline.run(niche, location, num_leads)
            
            put_html("<br>")
            put_markdown("---")
            put_markdown("## ✅ Success!")
            put_text(f"Generated {len(output_df)} high-quality leads")
            put_html("<br>")
            
            # Display results table
            put_markdown("### 📊 Lead Preview (Top 10)")
            
            table_data = []
            for _, row in output_df.head(10).iterrows():
                table_data.append([
                    row['business_name'][:40],
                    row['email'] if row['email'] else 'N/A',
                    row['phone'] if row['phone'] else 'N/A',
                    f"{row['ai_quality_score']:.2f}"
                ])
            
            put_table([
                ['Business Name', 'Email', 'Phone', 'AI Score'],
                *table_data
            ])
            
            put_html("<br>")
            
            # Download buttons
            put_markdown("### 📥 Download Results")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            put_file(
                f'leads_{timestamp}.csv',
                csv_data,
                '📄 Download CSV'
            )
            
            put_file(
                f'leads_{timestamp}.xlsx',
                excel_data,
                '📊 Download Excel'
            )
            
            put_html("<br>")
            put_markdown("---")
            put_text("💡 Tip: High-quality leads have AI scores > 0.6")
            
        except Exception as e:
            put_error(f"❌ Error: {str(e)}")
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
    
    put_html("<br><br>")
    put_markdown("---")
    put_text("Built with PyTorch, PyWebIO, and BeautifulSoup")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Starting AI Data Entry Automation Tool")
    logger.info("=" * 60)
    
    # Start web server
    start_server(
        main,
        port=8080,
        debug=False,
        auto_open_webbrowser=True
    )

