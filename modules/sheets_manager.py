"""
Google Sheets Integration Module
Handles connection and data operations with Google Sheets
"""
import json
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

from config import Config, SHEET_HEADERS


class GoogleSheetsManager:
    """Manages Google Sheets operations for storing assessment data"""
    
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    def __init__(self):
        self.client = None
        self.sheet = None
        self.worksheet = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Google Sheets"""
        try:
            # Get credentials from config
            creds_dict = Config.get_google_credentials()
            
            if not creds_dict:
                raise ValueError("Google service account credentials not found")
            
            # Create credentials object
            credentials = Credentials.from_service_account_info(
                creds_dict, 
                scopes=self.SCOPES
            )
            
            # Authorize and create client
            self.client = gspread.authorize(credentials)
            
            # Open the spreadsheet
            self.sheet = self.client.open_by_key(Config.GOOGLE_SHEET_ID)
            
            # Get or create the worksheet
            self._setup_worksheet()
            
            print("✓ Successfully connected to Google Sheets")
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to Google Sheets: {str(e)}")
            return False
    
    def _setup_worksheet(self):
        """Set up the worksheet with headers if needed"""
        try:
            # Try to get existing worksheet
            try:
                self.worksheet = self.sheet.worksheet("Assessment Data")
            except gspread.WorksheetNotFound:
                # Create new worksheet
                self.worksheet = self.sheet.add_worksheet(
                    title="Assessment Data",
                    rows=1000,
                    cols=len(SHEET_HEADERS)
                )
            
            # Check if headers exist
            first_row = self.worksheet.row_values(1)
            
            if not first_row or first_row[0] != SHEET_HEADERS[0]:
                # Add headers
                self.worksheet.update('A1', [SHEET_HEADERS])
                
                # Format headers (bold, background color)
                self.worksheet.format('A1:AZ1', {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.2, 'green': 0.4, 'blue': 0.8}
                })
                
                print("✓ Worksheet headers initialized")
            
        except Exception as e:
            print(f"✗ Error setting up worksheet: {str(e)}")
    
    def add_record(self, data_row):
        """Add a new record (row) to the sheet"""
        try:
            if not self.worksheet:
                raise ValueError("Worksheet not initialized")
            
            # Append the row
            self.worksheet.append_row(data_row, value_input_option='USER_ENTERED')
            
            print(f"✓ Record added successfully at {datetime.now()}")
            return True, "Record added successfully"
            
        except Exception as e:
            error_msg = f"Failed to add record: {str(e)}"
            print(f"✗ {error_msg}")
            return False, error_msg
    
    def get_all_records(self):
        """Retrieve all records from the sheet"""
        try:
            if not self.worksheet:
                raise ValueError("Worksheet not initialized")
            
            records = self.worksheet.get_all_records()
            return records
            
        except Exception as e:
            print(f"✗ Failed to get records: {str(e)}")
            return []
    
    def get_record_count(self):
        """Get the number of records in the sheet"""
        try:
            if not self.worksheet:
                return 0
            
            # Get all values and count non-empty rows (excluding header)
            all_values = self.worksheet.get_all_values()
            return max(0, len(all_values) - 1)
            
        except Exception as e:
            print(f"✗ Failed to get record count: {str(e)}")
            return 0
    
    def search_by_registration(self, reg_no):
        """Search for records by registration number"""
        try:
            if not self.worksheet:
                return []
            
            # Find all cells matching the registration number
            cells = self.worksheet.findall(reg_no)
            
            records = []
            for cell in cells:
                if cell.col == 2:  # Registration No is column 2
                    row_data = self.worksheet.row_values(cell.row)
                    records.append(row_data)
            
            return records
            
        except Exception as e:
            print(f"✗ Search failed: {str(e)}")
            return []
    
    def update_record(self, row_number, data_row):
        """Update an existing record"""
        try:
            if not self.worksheet:
                raise ValueError("Worksheet not initialized")
            
            # Update the entire row
            cell_range = f'A{row_number}:{chr(65 + len(data_row) - 1)}{row_number}'
            self.worksheet.update(cell_range, [data_row])
            
            return True, "Record updated successfully"
            
        except Exception as e:
            error_msg = f"Failed to update record: {str(e)}"
            print(f"✗ {error_msg}")
            return False, error_msg
    
    def delete_record(self, row_number):
        """Delete a record by row number"""
        try:
            if not self.worksheet:
                raise ValueError("Worksheet not initialized")
            
            self.worksheet.delete_rows(row_number)
            return True, "Record deleted successfully"
            
        except Exception as e:
            error_msg = f"Failed to delete record: {str(e)}"
            print(f"✗ {error_msg}")
            return False, error_msg
    
    def get_sheet_url(self):
        """Get the URL of the Google Sheet"""
        return f"https://docs.google.com/spreadsheets/d/{Config.GOOGLE_SHEET_ID}"
    
    def check_connection(self):
        """Check if connection to Google Sheets is active"""
        try:
            if self.worksheet:
                # Try a simple operation
                self.worksheet.row_values(1)
                return True
            return False
        except:
            return False


# Singleton instance
_sheets_manager = None

def get_sheets_manager():
    """Get or create the Google Sheets manager instance"""
    global _sheets_manager
    if _sheets_manager is None:
        _sheets_manager = GoogleSheetsManager()
    return _sheets_manager


# Test the module
if __name__ == "__main__":
    print("Testing Google Sheets connection...")
    manager = get_sheets_manager()
    
    if manager.check_connection():
        print(f"\n✓ Connection successful!")
        print(f"Sheet URL: {manager.get_sheet_url()}")
        print(f"Current record count: {manager.get_record_count()}")
        
        # Test adding a sample record
        test_data = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '113383106069',
            '1133',
            '09/10/2025',
            'BE/ECE',
            'VI B',
            'CEC367',
            'INDUSTRIAL IOT AND INDUSTRY 4.0',
            '02',
            'J. Mini',
        ]
        # Pad with empty values to match header length
        test_data.extend([''] * (len(SHEET_HEADERS) - len(test_data)))
        
        print("\nAdding test record...")
        success, message = manager.add_record(test_data)
        print(f"Result: {message}")
    else:
        print("\n✗ Connection failed. Please check your credentials.")
