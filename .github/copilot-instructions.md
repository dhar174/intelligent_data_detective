# Intelligent Data Detective - GitHub Copilot Instructions

**ALWAYS follow these instructions first and fallback to additional search and context gathering only if the information here is incomplete or found to be in error.**

## Working Effectively

Bootstrap, build, and test the repository:

```bash
# Install core dependencies (takes ~2-3 minutes)
pip install langchain langchain-core langchain-openai langchain_experimental langgraph
pip install pandas numpy scipy scikit-learn matplotlib seaborn
pip install pydantic python-dotenv tiktoken openpyxl xhtml2pdf
pip install tavily-python chromadb joblib

# Install development dependencies (takes ~30 seconds)
pip install pytest black flake8 mypy jupyter

# Run tests to validate setup (takes ~1 second)
python3 -m pytest test_intelligent_data_detective.py -v
# Expected: 22 tests pass

# Test error handling framework (takes ~1 second, 1 test may fail - this is acceptable)
python3 -m pytest test_error_handling_framework.py -v
# Expected: 15/16 tests pass (1 known failure in edge case)
```

**NEVER CANCEL** any long-running operations. Set timeout to 60+ minutes for full workflow execution.

## Core Architecture

This is a **Jupyter notebook-based multi-agent system** using LangChain and LangGraph:

- **Main implementation**: `IntelligentDataDetective_beta_v5.ipynb` (27 cells)
- **Multi-agent workflow**: Data Cleaner → Analyst → Visualization → Report Generator
- **Execution time**: 6-25 minutes for complete analysis (NEVER CANCEL)
- **API requirements**: OpenAI API key (required), Tavily API key (optional)

## Environment Setup

**Prerequisites:**
- Python 3.10+ (validated on 3.12.3)
- OpenAI API key for LLM operations
- Tavily API key (optional, for web search features)

**Required environment variables:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"  # Optional
```

## Running and Testing

**Primary execution method** - Jupyter notebook:
```bash
# Start Jupyter (takes ~5-10 seconds)
jupyter notebook IntelligentDataDetective_beta_v5.ipynb

# OR for JupyterLab
jupyter lab IntelligentDataDetective_beta_v5.ipynb
```

**Quick validation** (no API keys needed):
```bash
# Test core imports and basic functionality (takes ~3 seconds)
python3 -c "
import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
print('All core dependencies working correctly')
"
```

**Full workflow execution** (requires API keys):
- **Duration**: 6-8 minutes (small datasets), 12-15 minutes (medium), 20-25 minutes (large)
- **NEVER CANCEL**: Always wait for completion
- **Timeout setting**: Use 60+ minutes minimum

## Testing and Validation

**Run all tests:**
```bash
# Core functionality tests (takes ~1 second) - NEVER CANCEL
python3 -m pytest test_intelligent_data_detective.py -v
# Expected: 22 tests pass

# Error handling tests (takes ~1 second) - NEVER CANCEL  
python3 -m pytest test_error_handling_framework.py -v
# Expected: 15/16 tests pass (1 known failure is acceptable)

# Run all tests together (takes ~2 seconds) - NEVER CANCEL
python3 -m pytest -v
```

**Code quality checks:**
```bash
# Format code (takes ~1 second) - NEVER CANCEL
black test_intelligent_data_detective.py test_error_handling_framework.py

# Lint code (takes ~1 second) - NEVER CANCEL
flake8 test_intelligent_data_detective.py --max-line-length=88 --extend-ignore=E203,E501
# Note: Expect some whitespace warnings - these are acceptable
```

## Manual Validation Scenarios

**ALWAYS manually validate changes** by running through these complete scenarios:

### Scenario 1: Basic Data Analysis Workflow
```bash
# Test basic data operations (takes ~3 seconds) - NEVER CANCEL
python3 -c "
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create test data
df = pd.DataFrame({
    'values': np.random.randn(1000),
    'categories': np.random.choice(['A', 'B', 'C'], 1000)
})

# Basic analysis
summary = df.describe()
correlation = df.select_dtypes(include=[np.number]).corr()
grouped = df.groupby('categories').agg({'values': ['mean', 'std']})

# Create visualization
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='categories', y='values')
plt.savefig('/tmp/test_validation.png', dpi=150, bbox_inches='tight')
plt.close()

print('✅ Basic data analysis workflow validated')
"
```

### Scenario 2: Notebook Cell Execution (if API keys available)
1. Open `IntelligentDataDetective_beta_v5.ipynb`
2. Run cells 1-5 (setup and imports) - takes ~10-15 seconds - NEVER CANCEL
3. Verify no import errors
4. Check sample data loading works correctly
5. **For full validation**: Run complete workflow - takes 6-25 minutes - NEVER CANCEL

### Scenario 3: Test Pydantic Models and Data Structures
```bash
# Validate data models (takes ~1 second) - NEVER CANCEL
python3 -c "
from test_intelligent_data_detective import AnalysisConfig, CleaningMetadata

# Test model creation
config = AnalysisConfig(report_author='Test Author')
metadata = CleaningMetadata(
    steps_taken=['remove_duplicates', 'fill_missing'],
    data_description_after_cleaning='Clean dataset ready for analysis'
)

print('✅ Pydantic models validated')
"
```

## Critical Timing and Timeout Information

**NEVER CANCEL these operations:**

| Operation | Expected Time | Timeout Setting |
|-----------|---------------|-----------------|
| Dependency installation | 2-5 minutes | 10+ minutes |
| Test suite execution | 1-2 seconds | 60+ seconds |
| Basic data operations | <5 seconds | 60+ seconds |
| Jupyter notebook startup | 5-10 seconds | 120+ seconds |
| **Full workflow execution** | **6-25 minutes** | **60+ minutes** |
| Code formatting (black) | 1-3 seconds | 60+ seconds |
| Linting (flake8) | 1-3 seconds | 60+ seconds |

**CRITICAL**: The full multi-agent workflow can take up to 25 minutes for large datasets. This is NORMAL behavior - do not cancel or interrupt.

## Common Validation Steps

**Before making changes:**
1. Run `python3 -m pytest test_intelligent_data_detective.py -v` (22 tests should pass)
2. Test basic imports: `python3 -c "from langchain_core.messages import HumanMessage; print('OK')"`
3. Validate core data operations (see Scenario 1 above)

**After making changes:**
1. Run full test suite: `python3 -m pytest -v`
2. Check code formatting: `black --check .` (fix if needed)
3. Run manual validation scenarios
4. **For API-related changes**: Test notebook execution with real data

## Project Structure Reference

**Key files:**
- `IntelligentDataDetective_beta_v5.ipynb` - Main implementation (27 cells)
- `test_intelligent_data_detective.py` - Core functionality tests (22 tests)
- `test_error_handling_framework.py` - Error handling tests (16 tests)
- `README.md` - Project documentation and usage examples
- `complete_memory_integration_analysis.md` - Detailed workflow documentation

**Documentation files:**
- `IntelligentDataDetective_Documentation.md` - Notebook structure analysis
- `idd_v5_technical_review.md` - Technical implementation review
- `Project_Tech_Spec_Intelligent_Data_Detective.md` - Technical specification

**Generated artifacts:**
- `idd_v4_state_graph.mmd` - Mermaid diagram of agent workflow
- `idd_v4_state_graph.png` - Visual representation of state graph

## Known Issues and Workarounds

1. **Code formatting**: Test files need formatting fixes
   - Workaround: Run `black test_intelligent_data_detective.py` before committing
   - Expected: Some style warnings are acceptable

2. **One test failure** in error handling framework
   - File: `test_error_handling_framework.py`
   - Issue: Edge case in function signature handling
   - Impact: No functional impact on main system

3. **API key requirements**: 
   - Main notebook functionality requires OpenAI API key
   - Workaround: Use test scenarios without API calls for basic validation

## Multi-Agent Workflow Details

**Agent Types:**
- **Supervisor Agent**: Orchestrates workflow and routing decisions
- **Data Cleaner Agent**: Handles data quality, missing values, outliers
- **Analyst Agent**: Performs statistical analysis and pattern detection  
- **Visualization Agent**: Creates charts and graphs
- **Report Generator Agent**: Synthesizes findings into reports

**Typical Execution Flow:**
1. **Initial Analysis** (30-60 seconds)
2. **Data Cleaning** (60-120 seconds) 
3. **Statistical Analysis** (120-180 seconds)
4. **Visualization Generation** (90-150 seconds)
5. **Report Generation** (90-180 seconds)
6. **File Writing** (20-40 seconds)

**State Management:**
- Uses LangGraph StateGraph with memory persistence
- Checkpoint-based recovery for error resilience
- Streaming execution with real-time updates

## Validation Commands Summary

**Essential validation workflow:**
```bash
# 1. Install dependencies (2-5 minutes) - NEVER CANCEL
pip install langchain langchain-core langchain-openai langchain_experimental langgraph pandas numpy scipy scikit-learn matplotlib seaborn pydantic python-dotenv tiktoken openpyxl xhtml2pdf tavily-python chromadb joblib pytest black flake8 mypy jupyter

# 2. Run tests (1-2 seconds) - NEVER CANCEL  
python3 -m pytest test_intelligent_data_detective.py -v

# 3. Validate basic functionality (3 seconds) - NEVER CANCEL
python3 -c "import pandas as pd; from langchain_core.messages import HumanMessage; print('✅ Ready')"

# 4. Format code (1 second) - NEVER CANCEL
black test_intelligent_data_detective.py

# 5. Manual scenario testing (see scenarios above)
```

**For notebook changes**: Always test with `jupyter notebook IntelligentDataDetective_beta_v5.ipynb` and run at least the first 5 cells.

**For algorithm changes**: Always run complete manual validation scenarios and check that test suite still passes.

Remember: This system is designed for comprehensive data analysis workflows that naturally take time to complete. Patience during execution is essential for proper validation.