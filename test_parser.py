#!/usr/bin/env python3
"""
Test suite for generated bank statement parsers

This test script validates that generated parsers work correctly
and match the expected CSV output format.
"""

import pytest
import pandas as pd
import os
import sys
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

class ParserTester:
    """Test framework for bank statement parsers"""
    
    def __init__(self, bank_name: str):
        self.bank_name = bank_name.lower()
        self.parser_path = f"custom_parsers/{self.bank_name}_parser.py"
        self.pdf_path = f"data/{self.bank_name}/{self.bank_name}_sample.pdf"
        self.csv_path = f"data/{self.bank_name}/{self.bank_name}_sample.csv"
        self.parser_module = None
        self.expected_df = None
    
    def setup(self) -> bool:
        """Setup test environment and load required files"""
        print(f"Setting up tests for {self.bank_name.upper()} parser...")
        
        # Check if all required files exist
        missing_files = []
        if not os.path.exists(self.parser_path):
            missing_files.append(self.parser_path)
        if not os.path.exists(self.pdf_path):
            missing_files.append(self.pdf_path)
        if not os.path.exists(self.csv_path):
            missing_files.append(self.csv_path)
        
        if missing_files:
            print(f"Missing required files: {missing_files}")
            return False
        
        # Load expected CSV
        try:
            self.expected_df = pd.read_csv(self.csv_path)
            print(f"Expected CSV loaded: {len(self.expected_df)} rows, {len(self.expected_df.columns)} columns")
        except Exception as e:
            print(f"Failed to load expected CSV: {e}")
            return False
        
        # Import parser module
        try:
            self.parser_module = self._import_parser()
            if not self.parser_module:
                print("Failed to import parser module")
                return False
            print("Parser module imported successfully")
        except Exception as e:
            print(f"Failed to import parser: {e}")
            return False
        
        return True
    
    def _import_parser(self) -> Optional[Any]:
        """Dynamically import the generated parser"""
        try:
            spec = importlib.util.spec_from_file_location("parser_module", self.parser_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'parse'):
                print("Parser module does not have 'parse' function")
                return None
            
            return module
        except Exception as e:
            print(f"Import error: {e}")
            return None
    
    def test_parser_exists(self) -> bool:
        """Test that parser file exists and has parse function"""
        print("\n1. Testing parser file existence...")
        
        if not os.path.exists(self.parser_path):
            print(f"FAIL: Parser file not found: {self.parser_path}")
            return False
        
        if not self.parser_module or not hasattr(self.parser_module, 'parse'):
            print("FAIL: Parser module missing 'parse' function")
            return False
        
        print("PASS: Parser file exists with parse function")
        return True
    
    def test_parser_execution(self) -> tuple[bool, Optional[pd.DataFrame]]:
        """Test that parser executes without errors"""
        print("\n2. Testing parser execution...")
        
        try:
            result_df = self.parser_module.parse(self.pdf_path)
            
            if result_df is None:
                print("FAIL: Parser returned None")
                return False, None
            
            if not isinstance(result_df, pd.DataFrame):
                print(f"FAIL: Parser returned {type(result_df)}, expected DataFrame")
                return False, None
            
            print(f"PASS: Parser executed successfully, returned DataFrame with {len(result_df)} rows")
            return True, result_df
        
        except Exception as e:
            print(f"FAIL: Parser execution error: {e}")
            return False, None
    
    def test_schema_match(self, result_df: pd.DataFrame) -> bool:
        """Test that result DataFrame matches expected schema"""
        print("\n3. Testing schema compliance...")
        
        expected_cols = list(self.expected_df.columns)
        actual_cols = list(result_df.columns)
        
        if expected_cols != actual_cols:
            print(f"FAIL: Column mismatch")
            print(f"  Expected: {expected_cols}")
            print(f"  Actual:   {actual_cols}")
            
            missing = set(expected_cols) - set(actual_cols)
            extra = set(actual_cols) - set(expected_cols)
            
            if missing:
                print(f"  Missing columns: {missing}")
            if extra:
                print(f"  Extra columns: {extra}")
            
            return False
        
        print(f"PASS: Schema matches - {len(expected_cols)} columns in correct order")
        return True
    
    def test_data_extraction(self, result_df: pd.DataFrame) -> bool:
        """Test data extraction quality"""
        print("\n4. Testing data extraction quality...")
        
        issues = []
        
        # Check if DataFrame is empty
        if len(result_df) == 0:
            issues.append("DataFrame is empty - no transactions extracted")
        
        # Check row count reasonableness
        expected_rows = len(self.expected_df)
        actual_rows = len(result_df)
        
        if actual_rows < expected_rows * 0.5:
            issues.append(f"Too few rows: {actual_rows} vs expected ~{expected_rows}")
        elif actual_rows > expected_rows * 1.5:
            issues.append(f"Too many rows: {actual_rows} vs expected ~{expected_rows}")
        
        # Check for all-empty rows
        empty_rows = result_df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            issues.append(f"Found {empty_rows} completely empty rows")
        
        # Check date column for valid dates
        date_col = None
        for col in result_df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        if date_col:
            try:
                pd.to_datetime(result_df[date_col], errors='raise')
            except:
                issues.append(f"Invalid dates in {date_col} column")
        
        # Check amount columns
        amount_cols = [col for col in result_df.columns 
                      if any(word in col.lower() for word in ['amt', 'debit', 'credit', 'balance', 'amount'])]
        
        for col in amount_cols:
            non_empty_values = result_df[col][result_df[col] != '']
            if len(non_empty_values) > 0:
                try:
                    # Try to convert non-empty values to numeric
                    pd.to_numeric(non_empty_values, errors='raise')
                except:
                    issues.append(f"Invalid numeric values in {col}")
        
        if issues:
            print("FAIL: Data extraction issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print(f"PASS: Data extraction quality acceptable ({actual_rows} rows extracted)")
        return True
    
    def test_empty_value_handling(self, result_df: pd.DataFrame) -> bool:
        """Test proper handling of empty values in amount columns"""
        print("\n5. Testing empty value handling...")
        
        amount_cols = [col for col in result_df.columns 
                      if any(word in col.lower() for word in ['debit', 'credit']) and 'amt' in col.lower()]
        
        if not amount_cols:
            print("SKIP: No debit/credit amount columns found")
            return True
        
        issues = []
        
        for col in amount_cols:
            # Check for proper empty string handling
            empty_count = (result_df[col] == '').sum()
            null_count = result_df[col].isnull().sum()
            
            if null_count > 0:
                issues.append(f"Column {col} has {null_count} null values (should be empty strings)")
            
            if empty_count == 0:
                issues.append(f"Column {col} has no empty values (expected some transactions to have empty debit or credit)")
        
        if issues:
            print("FAIL: Empty value handling issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("PASS: Empty values properly handled as empty strings")
        return True
    
    def test_dataframe_equals(self, result_df: pd.DataFrame) -> bool:
        """Test if result DataFrame equals expected (strict comparison)"""
        print("\n6. Testing DataFrame equality (strict)...")
        
        try:
            # Ensure same column order
            result_df = result_df[self.expected_df.columns]
            
            # Convert both to string for comparison (handles mixed types)
            expected_str = self.expected_df.astype(str)
            result_str = result_df.astype(str)
            
            if expected_str.equals(result_str):
                print("PASS: DataFrames are identical")
                return True
            else:
                print("FAIL: DataFrames differ")
                
                # Show first few differences
                if len(result_df) != len(self.expected_df):
                    print(f"  Row count differs: {len(result_df)} vs {len(self.expected_df)}")
                else:
                    # Find differing cells
                    diff_mask = (expected_str != result_str)
                    diff_count = diff_mask.sum().sum()
                    print(f"  {diff_count} cells differ")
                    
                    # Show first few differences
                    for col in self.expected_df.columns:
                        col_diffs = diff_mask[col].sum()
                        if col_diffs > 0:
                            print(f"    {col}: {col_diffs} differences")
                            # Show first difference in this column
                            first_diff_idx = diff_mask[col].idxmax()
                            expected_val = expected_str.loc[first_diff_idx, col]
                            actual_val = result_str.loc[first_diff_idx, col]
                            print(f"      Row {first_diff_idx}: expected '{expected_val}', got '{actual_val}'")
                
                return False
                
        except Exception as e:
            print(f"FAIL: Error during comparison: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        if not self.setup():
            return {"setup": False}
        
        results = {}
        
        # Test 1: Parser exists
        results["parser_exists"] = self.test_parser_exists()
        
        # Test 2: Parser execution
        execution_success, result_df = self.test_parser_execution()
        results["parser_execution"] = execution_success
        
        if not execution_success or result_df is None:
            print("\nStopping tests - parser execution failed")
            return results
        
        # Test 3: Schema match
        results["schema_match"] = self.test_schema_match(result_df)
        
        # Test 4: Data extraction quality
        results["data_extraction"] = self.test_data_extraction(result_df)
        
        # Test 5: Empty value handling
        results["empty_value_handling"] = self.test_empty_value_handling(result_df)
        
        # Test 6: DataFrame equality
        results["dataframe_equals"] = self.test_dataframe_equals(result_df)
        
        return results
    
    def print_summary(self, results: Dict[str, bool]) -> None:
        """Print test summary"""
        print("\n" + "="*60)
        print(f"TEST SUMMARY - {self.bank_name.upper()} PARSER")
        print("="*60)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"{test_name.replace('_', ' ').title():<25} : {status}")
        
        print("-"*60)
        print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("SUCCESS: All tests passed!")
        else:
            print("FAILURE: Some tests failed")
        
        print("="*60)

def test_icici_parser():
    """Pytest function for ICICI parser"""
    tester = ParserTester("icici")
    results = tester.run_all_tests()
    tester.print_summary(results)
    
    # Assert all tests passed for pytest
    assert all(results.values()), f"Some tests failed: {results}"

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='Test bank statement parsers')
    parser.add_argument('--bank', required=True, help='Bank name to test (e.g., icici, sbi)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    tester = ParserTester(args.bank)
    results = tester.run_all_tests()
    tester.print_summary(results)
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()