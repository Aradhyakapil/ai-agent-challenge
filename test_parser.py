#!/usr/bin/env python3
"""
Test script to view the output of the generated parser
"""

import pandas as pd
import sys
import os

def test_parser():
    """Test the generated ICICI parser and display results"""
    
    try:
        # Import the generated parser
        sys.path.append('custom_parsers')
        from icici_parser import parse
        
        # Parse the PDF
        print("🔄 Parsing PDF...")
        pdf_path = "data/icici/icici_sample.pdf"
        result_df = parse(pdf_path)
        
        print(f"✅ Parsing successful!")
        print(f"📊 DataFrame shape: {result_df.shape}")
        print(f"📋 Columns: {list(result_df.columns)}")
        print("\n" + "="*80)
        
        # Display first 10 rows
        print("📄 First 10 transactions:")
        print(result_df.head(10).to_string(index=False))
        print("\n" + "="*80)
        
        # Display last 5 rows
        print("📄 Last 5 transactions:")
        print(result_df.tail(5).to_string(index=False))
        print("\n" + "="*80)
        
        # Show summary statistics
        print("📈 Summary Statistics:")
        
        # Convert amounts to numeric for statistics
        debit_col = result_df['Debit Amt'].replace('', '0').astype(float)
        credit_col = result_df['Credit Amt'].replace('', '0').astype(float)
        balance_col = result_df['Balance'].astype(float)
        
        print(f"Total Transactions: {len(result_df)}")
        print(f"Total Debits: ₹{debit_col.sum():,.2f}")
        print(f"Total Credits: ₹{credit_col.sum():,.2f}")
        print(f"Final Balance: ₹{balance_col.iloc[-1]:,.2f}")
        print(f"Date Range: {result_df['Date'].iloc[0]} to {result_df['Date'].iloc[-1]}")
        
        # Compare with expected CSV
        print("\n" + "="*80)
        print("🔍 Comparing with expected CSV:")
        
        expected_df = pd.read_csv("data/icici/icici_sample.csv")
        
        print(f"Expected rows: {len(expected_df)}")
        print(f"Parsed rows: {len(result_df)}")
        print(f"Columns match: {set(result_df.columns) == set(expected_df.columns)}")
        
        # Check if data matches (basic comparison)
        if len(result_df) == len(expected_df):
            print("✅ Row count matches!")
        else:
            print(f"⚠️ Row count differs: expected {len(expected_df)}, got {len(result_df)}")
            
        print("\n" + "="*80)
        print("💾 Saving parsed data to 'parsed_output.csv'")
        result_df.to_csv('parsed_output.csv', index=False)
        print("✅ Saved! You can open 'parsed_output.csv' to view all data.")
        
    except ImportError as e:
        print(f"❌ Could not import parser: {e}")
        print("Make sure the parser was generated successfully.")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure the PDF file exists at data/icici/icici_sample.pdf")
        
    except Exception as e:
        print(f"❌ Error testing parser: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parser()
