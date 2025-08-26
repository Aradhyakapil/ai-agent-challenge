# Agent-as-Coder: Bank Statement PDF Parser Generator

An autonomous LangGraph-based AI agent that analyzes bank statement PDFs and automatically generates custom Python parsers. The agent uses a structured Plan → Generate → Test → Fix workflow to create robust parsers that extract transaction data from PDF statements into structured DataFrames.

## Architecture

The agent implements a **LangGraph StateGraph** with five interconnected nodes:

```
[Plan] → [Generate] → [Test] → [Complete]
   ↓         ↓          ↓
[End] ← [Fix] ←────────┘
```

**Workflow Nodes:**
- **Plan**: Analyzes PDF structure and CSV schema to understand data patterns
- **Generate**: Uses Gemini LLM to create custom parsing code based on analysis
- **Test**: Validates generated parser against expected output schema
- **Fix**: Performs error analysis and prepares retry with improved context
- **Complete**: Finalizes successful parser generation

The agent maintains state across nodes and automatically retries up to 3 attempts with self-correction based on test failures.

## Quick Start

### Prerequisites
```bash
pip install langgraph google-generativeai pandas pdfplumber PyPDF2 pytest
export GEMINI_API_KEY="your_api_key_here"
```

### Directory Structure
```
AI-AGENT-CHALLENGE/
│
├── custom_parsers/              # Custom parsers for different document formats
│   ├── __pycache__/             # Auto-generated cache
│   └── icici_parser.py          # Parser for ICICI bank statement
│
├── data/icici/                  # Sample input and output data
│   ├── icici_sample.pdf         # Example input PDF (ICICI statement)
│   ├── icici_sample.csv         # Ground truth/reference CSV
│
├── agent.py                     # Main entry point to run the parsing agent
├── parsed_output.csv            # Generated output after parsing
├── test_parser1.py              # Test script for parser validation
├── README.md                    # Project documentation

```

### Generate Parser
```bash
python agent.py --target icici
```

### Test Generated Parser
```bash
python test_parser.py --bank icici
pytest test_parser.py::test_icici_parser -v
```

### Use Generated Parser
```python
from custom_parsers.icici_parser import parse
df = parse("path/to/statement.pdf")
```

## Usage Instructions

### Step 1: Prepare Data Files
Place your bank's sample PDF and corresponding CSV in the data directory:
```
data/{bank_name}/{bank_name}_sample.pdf
data/{bank_name}/{bank_name}_sample.csv
```

### Step 2: Set API Key
```bash
export GEMINI_API_KEY="your_gemini_api_key"
# Or pass directly: python agent.py --target icici --api-key "your_key"
```

### Step 3: Run Agent
```bash
python agent.py --target {bank_name}
```

### Step 4: Test Parser
```bash
python test_parser.py --bank {bank_name}
```

### Step 5: Deploy Parser
The generated `custom_parsers/{bank_name}_parser.py` contains a `parse(pdf_path: str) -> pd.DataFrame` function ready for production use.

## Technical Specifications

### Parser Contract
Generated parsers implement the following interface:
```python
def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parse bank statement PDF into structured DataFrame
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        pd.DataFrame with columns matching expected CSV schema
    """
```

### Output Requirements
- DataFrame columns must exactly match the reference CSV schema
- Empty debit/credit amounts represented as empty strings `''`
- Numeric values properly formatted and parseable
- All transactions from multi-page PDFs extracted
- Headers and footers automatically filtered out

### Error Handling
The agent implements robust error recovery:
- **Syntax Errors**: Code validation and import testing
- **Schema Mismatches**: Column name and order correction
- **Data Quality Issues**: Transaction extraction improvement
- **Edge Cases**: Multi-page handling and format variations

## Testing Framework

The included `test_parser.py` provides comprehensive validation:

- **Functional Testing**: Parser execution and error handling
- **Schema Compliance**: Column structure and data types
- **Data Quality**: Row counts and value validation  
- **Edge Case Handling**: Empty values and numeric conversion
- **Strict Equality**: DataFrame.equals() comparison with expected output

## Agent Autonomy Features

### Self-Debugging Loop
The agent automatically:
1. Detects test failures and extracts error details
2. Analyzes failure patterns and root causes
3. Generates improved prompts with specific fix guidance
4. Retries generation with enhanced context

### Adaptive Prompting
- PDF structure analysis guides parser generation
- Error-specific feedback improves subsequent attempts
- Bank-agnostic approach works across different statement formats
- Transaction pattern recognition for robust extraction

### Quality Assurance
- Multi-method PDF text extraction (pdfplumber + PyPDF2)
- Comprehensive output validation
- Schema enforcement and type checking
- Automatic retry with progressive refinement

## Dependencies

- `langgraph`: Workflow orchestration and state management
- `google-generativeai`: LLM integration for code generation
- `pandas`: DataFrame operations and CSV handling
- `pdfplumber`: Primary PDF text extraction
- `PyPDF2`: Fallback PDF extraction method
- `pytest`: Testing framework integration

## API Configuration

The agent supports Google Gemini API for code generation. Set your API key via environment variable or command line parameter. Free API credits available through Google AI Studio.

## Extensibility

To add support for new banks:
1. Create `data/{bank_name}/{bank_name}_sample.pdf` and corresponding CSV
2. Run `python agent.py --target {bank_name}`
3. The agent automatically adapts to new formats without code changes

The architecture is designed to handle diverse bank statement formats through dynamic analysis rather than hardcoded parsing rules.
