#!/usr/bin/env python3
"""
Agent-as-Coder: LangGraph-based Bank Statement PDF Parser Agent

This agent uses LangGraph to create a structured workflow that analyzes 
bank statement PDFs and CSVs to automatically generate custom parsers.

Architecture: LangGraph StateGraph with nodes for Plan → Generate → Test → Fix
"""

import argparse
import os
import sys
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict, Annotated
import google.generativeai as genai
import PyPDF2
import pdfplumber
import re
import traceback
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    """LangGraph state schema for the agent workflow"""
    target_bank: str
    pdf_path: str
    csv_path: str
    parser_path: str
    pdf_content: str
    csv_schema: Dict[str, Any]
    current_attempt: int
    max_attempts: int
    last_error: str
    generated_code: str
    test_passed: bool
    workflow_complete: bool
    messages: Annotated[List[str], "Workflow messages"]

class BankStatementParserAgent:
    """
    LangGraph-based AI Agent for generating bank statement PDF parsers
    
    Uses a structured workflow: Plan → Generate → Test → Fix loop
    """
    
    def __init__(self, api_key: str):
        """Initialize agent with Gemini API and LangGraph"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.memory = MemorySaver()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow with nodes and edges"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("generate", self._generate_node) 
        workflow.add_node("test", self._test_node)
        workflow.add_node("fix", self._fix_node)
        workflow.add_node("complete", self._complete_node)
        
        # Set entry point
        workflow.set_entry_point("plan")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "plan",
            self._plan_router,
            {
                "generate": "generate",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "generate", 
            self._generate_router,
            {
                "test": "test",
                "fix": "fix"
            }
        )
        
        workflow.add_conditional_edges(
            "test",
            self._test_router,
            {
                "complete": "complete",
                "fix": "fix"
            }
        )
        
        workflow.add_conditional_edges(
            "fix",
            self._fix_router,
            {
                "generate": "generate",
                "end": END
            }
        )
        
        workflow.add_edge("complete", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def run(self, target_bank: str) -> bool:
        """
        Execute the LangGraph workflow for parser generation
        
        Args:
            target_bank: Name of target bank
            
        Returns:
            bool: Success status
        """
        # Initialize state
        initial_state: AgentState = {
            "target_bank": target_bank.lower(),
            "pdf_path": f"data/{target_bank.lower()}/{target_bank.lower()}_sample.pdf",
            "csv_path": f"data/{target_bank.lower()}/{target_bank.lower()}_sample.csv", 
            "parser_path": f"custom_parsers/{target_bank.lower()}_parser.py",
            "pdf_content": "",
            "csv_schema": {},
            "current_attempt": 0,
            "max_attempts": 3,
            "last_error": "",
            "generated_code": "",
            "test_passed": False,
            "workflow_complete": False,
            "messages": []
        }
        
        print(f"🚀 Starting LangGraph agent for {target_bank.upper()} bank statement parser")
        
        # Execute workflow
        config = {"configurable": {"thread_id": f"{target_bank}_parser"}}
        
        try:
            final_state = None
            for state in self.workflow.stream(initial_state, config):
                # Print current node execution
                node_name = list(state.keys())[0]
                current_state = list(state.values())[0]
                
                if current_state.get("messages"):
                    print(f"📍 {node_name.upper()}: {current_state['messages'][-1]}")
                
                final_state = current_state
                
                # Check if workflow completed successfully
                if current_state.get("workflow_complete"):
                    break
            
            if final_state and final_state.get("test_passed"):
                print(f"✅ Successfully created {final_state['parser_path']}")
                return True
            else:
                print("❌ Failed to generate working parser")
                return False
                
        except Exception as e:
            print(f"💥 Workflow execution error: {e}")
            return False
    
    def _plan_node(self, state: AgentState) -> AgentState:
        """Node 1: Plan - Analyze PDF and CSV structure"""
        try:
            state["messages"].append("Analyzing PDF and CSV structure...")
            
            # Read PDF content
            if not os.path.exists(state["pdf_path"]):
                state["last_error"] = f"PDF file not found: {state['pdf_path']}"
                return state
                
            state["pdf_content"] = self._extract_pdf_text(state["pdf_path"])
            if not state["pdf_content"].strip():
                state["last_error"] = "Could not extract text from PDF"
                return state
            
            # Read CSV schema
            if not os.path.exists(state["csv_path"]):
                state["last_error"] = f"CSV file not found: {state['csv_path']}"
                return state
                
            sample_df = pd.read_csv(state["csv_path"])
            state["csv_schema"] = {
                "columns": list(sample_df.columns),
                "dtypes": {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
                "sample_rows": sample_df.head(5).to_dict('records'),
                "total_rows": len(sample_df)
            }
            
            state["messages"].append(f"PDF: {len(state['pdf_content'])} chars, CSV: {len(sample_df)} rows")
            
        except Exception as e:
            state["last_error"] = f"Plan phase error: {str(e)}"
            state["messages"].append(f"Plan failed: {str(e)}")
        
        return state
    
    def _generate_node(self, state: AgentState) -> AgentState:
        """Node 2: Generate - Create parser code using LLM"""
        try:
            state["current_attempt"] += 1
            state["messages"].append(f"Generating parser code (attempt {state['current_attempt']})...")
            
            prompt = self._build_generation_prompt(state)
            response = self.model.generate_content(prompt)
            
            # Extract Python code from response
            if hasattr(response, "text"):
                resp_text = response.text
            elif hasattr(response, "candidates") and len(response.candidates) > 0:
                resp_text = response.candidates[0].content.parts[0].text
            else:
                resp_text = str(response)
            
            # Extract code block
            code_match = re.search(r'```python\n(.*?)```', resp_text, re.DOTALL)
            if code_match:
                code_text = code_match.group(1)
            elif 'def parse(' in resp_text:
                code_text = resp_text
            else:
                state["last_error"] = "No valid Python code found in LLM response"
                return state
            
            # Create complete parser file
            complete_code = self._build_complete_parser(code_text, state)
            
            # Save parser file
            os.makedirs(os.path.dirname(state["parser_path"]), exist_ok=True)
            with open(state["parser_path"], 'w', encoding='utf-8') as f:
                f.write(complete_code)
            
            state["generated_code"] = complete_code
            state["messages"].append("Single parser file generated and saved")
            
        except Exception as e:
            state["last_error"] = f"Generate phase error: {str(e)}"
            state["messages"].append(f"Generation failed: {str(e)}")
        
        return state
    
    def _test_node(self, state: AgentState) -> AgentState:
        """Node 3: Test - Validate generated parser"""
        try:
            state["messages"].append("Testing generated parser...")
            
            # Import and test parser
            parser_module = self._import_parser(state["parser_path"])
            if not parser_module:
                state["last_error"] = "Failed to import generated parser"
                return state
            
            # Execute parser with actual PDF path
            result_df = parser_module.parse(state["pdf_path"])
            
            # Validate output
            validation_result = self._validate_parser_output(result_df, state)
            if validation_result["valid"]:
                state["test_passed"] = True
                state["messages"].append("Parser test passed!")
            else:
                state["test_passed"] = False
                state["last_error"] = validation_result["error"]
                state["messages"].append(f"Test failed: {validation_result['error']}")
            
        except Exception as e:
            state["test_passed"] = False
            state["last_error"] = f"Test execution error: {str(e)}\n{traceback.format_exc()}"
            state["messages"].append(f"Test error: {str(e)}")
        
        return state
    
    def _fix_node(self, state: AgentState) -> AgentState:
        """Node 4: Fix - Analyze errors and prepare for retry"""
        try:
            state["messages"].append(f"Analyzing failure for retry (attempt {state['current_attempt']})...")
            
            # Add error analysis context for next generation
            error_analysis = self._analyze_error(state)
            state["last_error"] = f"{state['last_error']}\n\nError Analysis: {error_analysis}"
            
            state["messages"].append("Error analysis complete, preparing retry...")
            
        except Exception as e:
            state["messages"].append(f"Fix analysis error: {str(e)}")
        
        return state
    
    def _complete_node(self, state: AgentState) -> AgentState:
        """Node 5: Complete - Finalize successful workflow"""
        state["workflow_complete"] = True
        state["messages"].append("Workflow completed successfully!")
        return state
    
    # --- Routers ---
    def _plan_router(self, state: AgentState) -> str:
        """Route from plan node"""
        if state["last_error"]:
            return "end"
        return "generate"
    
    def _generate_router(self, state: AgentState) -> str:
        """Route from generate node"""
        if state["last_error"]:
            return "fix"
        return "test"
    
    def _test_router(self, state: AgentState) -> str:
        """Route from test node"""
        if state["test_passed"]:
            return "complete"
        return "fix"
    
    def _fix_router(self, state: AgentState) -> str:
        """Route from fix node"""
        if state["current_attempt"] >= state["max_attempts"]:
            return "end"
        return "generate"
    
    def _build_generation_prompt(self, state: AgentState) -> str:
        """Build comprehensive LLM prompt for parser generation"""
        error_context = ""
        if state["current_attempt"] > 1:
            error_context = (
                f"\nPREVIOUS ATTEMPT {state['current_attempt']-1} FAILED:\n"
                f"Error: {state['last_error']}\n\n"
                "Fix these issues: 1) use pdf_path parameter, 2) exact column names, "
                "3) preserve empty strings for missing amounts, 4) handle multi-page PDFs, "
                "5) robust regex patterns.\n\n"
            )
        
        pdf_sample = state["pdf_content"][:2000]
        structure_hints = self._analyze_pdf_structure(state["pdf_content"])
        
        prompt = (
            "You are an expert Python developer creating a bank statement PDF parser.\n\n"
            f"{error_context}"
            f"Bank: {state['target_bank'].upper()}\n"
            f"PDF Structure: {structure_hints}\n\n"
            "PDF Content Sample:\n"
            f"{pdf_sample}\n\n"
            "Expected Output Schema:\n"
            f"Columns: {state['csv_schema'].get('columns', [])}\n"
            f"Sample Data: {json.dumps(state['csv_schema'].get('sample_rows', [])[:3], indent=2)}\n\n"
            "REQUIREMENTS:\n"
            "1. Function signature: def parse(pdf_path: str) -> pd.DataFrame\n"
            "2. Use pdf_path parameter to open/read PDF - NEVER hardcode filenames\n"
            f"3. Return DataFrame with exact columns: {state['csv_schema'].get('columns', [])}\n"
            "4. For empty Debit/Credit amounts use empty string '' (not NaN/None/0)\n"
            "5. Handle multi-page PDFs and skip headers\n"
            "6. Include all imports (pandas, pdfplumber, re, etc.)\n"
            "7. Be robust to different transaction line formats\n\n"
            "Generate a complete, self-contained Python parser function."
        )
        return prompt
    
    def _build_complete_parser(self, generated_code: str, state: AgentState) -> str:
        """Build complete parser file with imports and validation"""
        expected_cols = state["csv_schema"].get("columns", [])
        
        # Clean generated code - remove any hardcoded filenames
        clean_code = re.sub(r'(["\'])([^"\']*\.pdf)\1', 'pdf_path', generated_code)
        
        complete_parser = f'''"""
Auto-generated parser for {state['target_bank'].upper()} bank statements
Generated by LangGraph Agent
"""
import pandas as pd
import pdfplumber
import re
import PyPDF2
from typing import Optional

{clean_code}

def _validate_output(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure output matches expected schema"""
    expected_columns = {expected_cols}
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Parser must return a pandas DataFrame")
    
    # Add missing columns with empty strings
    for col in expected_columns:
        if col not in df.columns:
            df[col] = ''
    
    # Reorder to expected columns
    df = df[expected_columns]
    
    # Handle amount columns - preserve empty strings
    amount_cols = [col for col in expected_columns 
                  if any(word in col.lower() for word in ['amt', 'debit', 'credit', 'balance', 'amount'])]
    
    for col in amount_cols:
        df[col] = df[col].astype(str).replace(['nan', 'None', 'NaN'], '').fillna('')
    
    return df

# Wrap the generated parse function to ensure validation
_original_parse = parse

def parse(pdf_path: str) -> pd.DataFrame:
    """Main parse function with validation"""
    result = _original_parse(pdf_path)
    return _validate_output(result)
'''
        return complete_parser
    
    def _analyze_pdf_structure(self, pdf_content: str) -> str:
        """Analyze PDF structure to provide parsing hints"""
        if not pdf_content:
            return "No content extracted"
        
        hints = []
        
        # Date format detection
        if re.search(r'\d{2}-\d{2}-\d{4}', pdf_content):
            hints.append("DD-MM-YYYY format")
        elif re.search(r'\d{2}/\d{2}/\d{4}', pdf_content):
            hints.append("DD/MM/YYYY format")
        
        # Amount detection
        if re.search(r'\d+\.\d{2}', pdf_content):
            hints.append("Decimal amounts present")
        
        # Column structure
        if 'Debit' in pdf_content and 'Credit' in pdf_content:
            hints.append("Separate Debit/Credit columns")
        
        # Transaction count estimation
        date_matches = len(re.findall(r'\d{2}[-/]\d{2}[-/]\d{4}', pdf_content))
        if date_matches > 0:
            hints.append(f"~{date_matches} transactions")
        
        return "; ".join(hints) if hints else "Standard format"
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        try:
            # Try pdfplumber first
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception:
            try:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                print(f"PDF extraction failed: {e}")
        
        return text.strip()
    
    def _import_parser(self, parser_path: str) -> Optional[Any]:
        """Dynamically import generated parser"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("generated_parser", parser_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'parse'):
                return None
            return module
        except Exception as e:
            print(f"Import error: {e}")
            return None
    
    def _validate_parser_output(self, result_df: pd.DataFrame, state: AgentState) -> Dict[str, Any]:
        """Validate parser output against expected schema"""
        try:
            if not isinstance(result_df, pd.DataFrame):
                return {"valid": False, "error": "Output is not a DataFrame"}
            
            if len(result_df) == 0:
                return {"valid": False, "error": "DataFrame is empty - no transactions extracted"}
            
            expected_cols = set(state["csv_schema"]["columns"])
            actual_cols = set(result_df.columns)
            
            if expected_cols != actual_cols:
                missing = expected_cols - actual_cols
                extra = actual_cols - expected_cols
                return {"valid": False, "error": f"Column mismatch. Missing: {missing}, Extra: {extra}"}
            
            # Check for reasonable number of rows
            expected_rows = state["csv_schema"]["total_rows"]
            if expected_rows and len(result_df) < expected_rows * 0.5:
                return {"valid": False, "error": f"Too few rows: {len(result_df)} vs expected ~{expected_rows}"}
            
            # Validate amount columns
            amount_cols = [col for col in result_df.columns 
                          if any(word in col.lower() for word in ['amt', 'debit', 'credit', 'balance'])]
            
            for col in amount_cols:
                non_empty = result_df[col][result_df[col] != '']
                if len(non_empty) > 0:
                    try:
                        pd.to_numeric(non_empty.replace('', '0'), errors='raise')
                    except:
                        return {"valid": False, "error": f"Invalid numeric values in {col}"}
            
            return {"valid": True, "error": ""}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def _analyze_error(self, state: AgentState) -> str:
        """Analyze error to provide fix guidance"""
        error = state.get("last_error", "")
        
        if "No such file" in error:
            return "Use pdf_path parameter instead of hardcoded filename"
        elif "Column mismatch" in error:
            return "Ensure exact column names match expected schema"
        elif "empty" in error.lower():
            return "Check PDF text extraction and transaction parsing logic"
        elif "numeric" in error.lower():
            return "Fix amount parsing - preserve empty strings for missing values"
        elif "few rows" in error:
            return "Improve transaction detection - check multi-page handling and regex patterns"
        else:
            return "Review PDF structure and adjust parsing logic accordingly"

def main():
    """CLI entry point for LangGraph agent"""
    parser = argparse.ArgumentParser(description='LangGraph Bank Statement PDF Parser Agent')
    parser.add_argument('--target', required=True, 
                       help='Target bank name (e.g., icici, sbi)')
    parser.add_argument('--api-key', 
                       default=os.getenv('GEMINI_API_KEY'),
                       help='Gemini API key (or set GEMINI_API_KEY env var)')
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("❌ Error: Gemini API key required. Set GEMINI_API_KEY env var or use --api-key")
        sys.exit(1)
    
    # Validate input files exist
    pdf_path = f"data/{args.target.lower()}/{args.target.lower()}_sample.pdf"
    csv_path = f"data/{args.target.lower()}/{args.target.lower()}_sample.csv"
    
    if not os.path.exists(pdf_path) or not os.path.exists(csv_path):
        print(f"❌ Error: Required files not found:")
        print(f"  PDF: {pdf_path}")
        print(f"  CSV: {csv_path}")
        sys.exit(1)
    
    # Run LangGraph agent
    agent = BankStatementParserAgent(args.api_key)
    success = agent.run(args.target)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()