# Intelligent Data Detective Notebook Documentation

## 1. Introduction
This document provides a comprehensive overview of the `IntelligentDataDetective_beta_v5.ipynb` notebook. The notebook implements an intelligent data analysis system using a multi-agent architecture built with LangGraph. It aims to automate various stages of data analysis, including data loading, cleaning, exploratory data analysis (EDA), visualization, and report generation. Each agent specializes in a specific task, and a supervisor agent coordinates the overall workflow.

## 2. Notebook Structure (Cell-by-Cell Analysis)

The notebook is structured into several cells, each with a specific role in setting up and running the data analysis pipeline.

*   **Cell 1 (ID: `nvqEP-kxAmiX`)**: Environment setup (Colab detection, API key retrieval from Colab `userdata` or environment variables) and installation of all required Python packages (e.g., `langchain`, `pandas`, `pydantic`, `matplotlib`, `seaborn`, `tavily-python`, `kagglehub`).
*   **Cell 2 (ID: `essential_imports_cell`)**: Imports all essential Python modules and specific components from libraries like `langchain`, `pandas`, `pydantic`, `numpy`, `os`, `typing`, etc., required for the entire notebook's functionality.
*   **Cell 3 (ID: `working_directory_definition_cell`)**: Defines the global `WORKING_DIRECTORY` variable using a temporary directory for file operations during the notebook's execution.
*   **Cell 4 (ID: `Y1NHB1z5BNWo`)**: Defines core Pydantic models for data structures and configuration (`AnalysisConfig`, `CleaningMetadata`, `InitialDescription`, `AnalysisInsights`, `VisualizationResults`, `ReportResults`, `DataQueryParams`, `CellIdentifier`, `GetDataParams`). It also defines the `DataFrameRegistry` class for managing pandas DataFrames (including an LRU cache mechanism) and instantiates a global instance `global_df_registry`. Crucially, this cell defines the main `State` class (inheriting from `AgentState`) that serves as the shared memory and state representation for the LangGraph workflow.
*   **Cell 5 (ID: `prompt_templates_cell`)**: Contains definitions for various `PromptTemplate` instances. These templates are used to guide the behavior of the different Langchain agents by providing specific instructions, context, and expected output formats. Templates include `data_cleaner_prompt_template`, `analyst_prompt_template_initial`, `analyst_prompt_template_main`, `file_writer_prompt_template`, `visualization_prompt_template`, and `report_generator_prompt_template`.
*   **Cell 6 (ID: `tool_definitions_cell`)**: Defines a comprehensive suite of custom tools for agents using the `@tool` decorator. These tools cover a wide range of functionalities including data loading, schema inspection, data cleaning operations (handling missing values, duplicates, type conversion), data transformation, various analysis tasks (descriptive statistics, correlation, hypothesis testing, outlier detection, normality testing), file I/O operations, Python code execution via REPL, web searches using Tavily API, and generation of visualizations (histograms, scatter plots, heatmaps, box plots) and reports (HTML, Markdown, PDF). Tools are grouped into lists (e.g., `data_cleaning_tools`, `analyst_tools`) for assignment to specific agents.
*   **Cell 7 (ID: `kBguFhuoZV5G`)**: Contains factory functions (e.g., `create_data_cleaner_agent`, `create_analyst_agent`) to create and configure instances of the various specialized agents. It also defines the `make_supervisor_node` function, which creates the supervisor agent responsible for orchestrating the workflow among the other agents.
*   **Cell 8 (ID: `oSLpBrTN3FIy`)**: Handles data loading. It downloads a sample dataset ("datafiniti/consumer-reviews-of-amazon-products") using `kagglehub`, reads the CSV data into a pandas DataFrame, and registers this DataFrame with the `global_df_registry`. It then instantiates all the defined agents, passing the necessary `df_id` and initial prompts.
*   **Cell 9 (ID: `ixJqmfH0lBfE`)**: Defines the node functions for the LangGraph graph. Each function acts as a wrapper around an agent's execution, managing input/output, state updates (like setting completion flags and storing results), and instructing the graph to return control to the supervisor.
*   **Cell 10 (ID: `8v52mP2CxPj4`)**: Sets up the LangGraph structure. It instantiates `StateGraph(State)`, adds the supervisor node and all agent nodes, and defines the edges that control the flow of execution (e.g., `START` to supervisor, all agents back to supervisor). The graph is then compiled with a `MemorySaver` checkpointer and an `InMemoryStore`.
*   **Cell 11 (ID: `WFTXS7s3x4Y3`)**: Visualizes the compiled graph structure using `data_detective_graph.get_graph().draw_mermaid_png()` and displays it as an image.
*   **Cell 12 (ID: `combined_streaming_cells`)**: Demonstrates the primary method of executing the graph using `data_detective_graph.stream()`. It shows how to provide initial inputs (messages, user prompt, config, DataFrame IDs) and how to iterate through and print the streaming updates from the graph execution. It also shows retrieval of the graph's state history.
*   **Cell 13 (ID: `streaming_orchestration_cell`)**: Defines an alternative `async` function `run_data_detective_stream` to encapsulate the graph execution and streaming logic, providing a reusable pattern for running the analysis.
*   **Cell 14 (ID: `LBK-Q-qH9yas`)**: Shows how to access and print specific information (like `analysis_insights` and `report_results`) from the final state snapshot of the graph after execution.
*   **Cell 15 (ID: `pydantic_test_cell`)**: Includes test cases for the `GetDataParams` Pydantic model to verify its validation logic, particularly for the `index` field.
*   **Cell 16 (ID: `unittest_dataframe_registry_cell`)**: Contains `unittest` test cases for the `DataFrameRegistry` class, ensuring its functionalities like DataFrame registration, retrieval, removal, LRU caching, and loading from raw paths work correctly.
*   **Cell 17 (ID: `ZgRwWoBT3bpU`)**: Contains additional imports such as `functools.cache`, `io.BytesIO`, and `ipywidgets`, some of which might be for features not fully utilized in the main agent flow or for potential interactive UI elements.

## 3. Core Data Structures

### 3.1. Pydantic Models
This section details the Pydantic models used for structuring data, configuration, and state within the Intelligent Data Detective.

#### `AnalysisConfig`
*   **Overall Purpose:** User-configurable settings for the data analysis workflow.
*   **Fields:**
    *   `default_visualization_style`: **Type:** `str`, **Purpose:** Default style for matplotlib/seaborn visualizations. **Default:** `"seaborn-v0_8-whitegrid"`
    *   `report_author`: **Type:** `Optional[str]`, **Purpose:** Author name to include in generated reports. **Default:** `None`
    *   `datetime_format_preference`: **Type:** `str`, **Purpose:** Preferred format for datetime string representations. **Default:** `"%Y-%m-%d %H:%M:%S"`
    *   `large_dataframe_preview_rows`: **Type:** `int`, **Purpose:** Number of rows for previewing large dataframes. **Default:** `5`

#### `CleaningMetadata`
*   **Overall Purpose:** Metadata about the data cleaning actions taken.
*   **Fields:**
    *   `steps_taken`: **Type:** `list[str]`, **Purpose:** List of cleaning steps performed. (Required)
    *   `data_description_after_cleaning`: **Type:** `str`, **Purpose:** Brief description of the dataset after cleaning. (Required)

#### `InitialDescription`
*   **Overall Purpose:** Initial description of the dataset.
*   **Fields:**
    *   `dataset_description`: **Type:** `str`, **Purpose:** Brief description of the dataset. (Required)
    *   `data_sample`: **Type:** `Optional[str]`, **Purpose:** Sample of the data (first few rows). **Default:** `None`

#### `AnalysisInsights`
*   **Overall Purpose:** Insights from the exploratory data analysis.
*   **Fields:**
    *   `summary`: **Type:** `str`, **Purpose:** Overall summary of EDA findings. (Required)
    *   `correlation_insights`: **Type:** `str`, **Purpose:** Key correlation insights identified. (Required)
    *   `anomaly_insights`: **Type:** `str`, **Purpose:** Anomalies or interesting patterns detected. (Required)
    *   `recommended_visualizations`: **Type:** `list[str]`, **Purpose:** List of recommended visualizations to illustrate findings. (Required)
    *   `recommended_next_steps`: **Type:** `Optional[List[str]]`, **Purpose:** List of recommended next analysis steps or questions to investigate based on the findings. **Default:** `None`

#### `VisualizationResults`
*   **Overall Purpose:** Results from the visualization generation.
*   **Fields:**
    *   `visualizations`: **Type:** `List[dict]`, **Purpose:** List of visualizations generated. Each dictionary should have the plot type and the base64 encoded image. (Required)

#### `ReportResults`
*   **Overall Purpose:** Results from the report generation.
*   **Fields:**
    *   `report_path`: **Type:** `str`, **Purpose:** Path to the generated report file. (Required)

#### `DataQueryParams`
*   **Overall Purpose:** Parameters for querying the DataFrame.
*   **Fields:**
    *   `columns`: **Type:** `List[str]`, **Purpose:** List of columns to include in the output. (Required)
    *   `filter_column`: **Type:** `Optional[str]`, **Purpose:** Column to apply the filter on. **Default:** `None`
    *   `filter_value`: **Type:** `Optional[str]`, **Purpose:** Value to filter the rows by. **Default:** `None`
    *   `operation`: **Type:** `str`, **Purpose:** Operation to perform: 'select', 'sum', 'mean', 'count', 'max', 'min', 'median', etc. **Default:** `"select"`

#### `CellIdentifier`
*   **Overall Purpose:** Identifies a single cell by row index and column name.
*   **Fields:**
    *   `row_index`: **Type:** `int`, **Purpose:** Row index of the cell. (Required)
    *   `column_name`: **Type:** `str`, **Purpose:** Column name of the cell. (Required)

#### `GetDataParams`
*   **Overall Purpose:** Parameters for retrieving data from the DataFrame.
*   **Fields:**
    *   `df_id`: **Type:** `str`, **Purpose:** DataFrame ID in the global registry. (Required)
    *   `index`: **Type:** `Union[int, List[int], Tuple[int, int]]`, **Purpose:** Specifies the rows to retrieve. Can be: 1) A single integer for one row. 2) A list of integers for multiple specific rows. 3) A 2-element tuple `(start, end)` for a range of rows (inclusive). (Required)
    *   `columns`: **Type:** `Union[str, List[str]]`, **Purpose:** A string (single column), a list of strings (multiple columns), or 'all' for all columns. **Default:** `"all"`
    *   `cells`: **Type:** `Optional[List[CellIdentifier]]`, **Purpose:** A list of cell identifier objects, each specifying a 'row_index' and 'column_name'. **Default:** `None`
*   **Validators:**
    *   `validate_index` (`@model_validator(mode='before')`): Validates `index` type (must be `int`, `list`, or `tuple`), tuple length (must be 2), and list elements (must be `int`).

### 3.2. `DataFrameRegistry` Class
The `DataFrameRegistry` class manages pandas DataFrames, acting as a central registry and LRU cache.

*   **Purpose:** To store, retrieve, and manage DataFrames and their metadata (like file paths) efficiently. It uses an LRU cache for quick access to frequently used DataFrames and supports lazy loading from disk.
*   **Initialization (`__init__`):**
    *   `capacity` (Optional\[`int`], default: `20`): Max DataFrames in the LRU cache.
    *   Internal structures:
        *   `registry` (`Dict[str, dict]`): Stores DataFrame objects and their raw paths, keyed by `df_id`.
        *   `df_id_to_raw_path` (`Dict[str, str]`): Maps `df_id` to `raw_path`.
        *   `cache` (`OrderedDict`): LRU cache for DataFrame objects.
*   **Methods:**
    *   `register_dataframe(self, df=None, df_id=None, raw_path="")`:
        *   Registers a DataFrame. Generates a UUID for `df_id` if not provided.
        *   Defaults `raw_path` to `WORKING_DIRECTORY / f"{df_id}.csv"` if empty.
        *   Stores DataFrame and path in `registry` and `df_id_to_raw_path`.
        *   Adds `df` to `cache` (with LRU eviction if capacity is hit).
        *   Returns: `df_id` (`str`).
    *   `get_dataframe(self, df_id: str, load_if_not_exists=False)`:
        *   Retrieves a DataFrame. Checks cache first (updates LRU order).
        *   If not in cache, checks `registry`. If `df` object is present, caches and returns it.
        *   If `df` is not in memory, `load_if_not_exists` is `True`, and `raw_path` exists, it loads the DataFrame from the CSV file, updates registry and cache.
        *   Handles `FileNotFoundError`.
        *   Returns: `Optional[pd.DataFrame]`.
    *   `remove_dataframe(self, df_id: str)`:
        *   Removes the DataFrame from `registry`, `cache`, and `df_id_to_raw_path`.
    *   `get_raw_path_from_id(self, df_id: str)`:
        *   Returns the `raw_path` for a given `df_id`.
        *   Returns: `Optional[str]`.
*   **Global Instance:** `global_df_registry = DataFrameRegistry()` is created for system-wide use.

### 3.3. `State` Model (`State` Class)
The `State` class defines the shared structure passed between nodes in the LangGraph workflow.

*   **Purpose:** To hold all information agents need, track task completion, and manage the flow of data and messages.
*   **Inheritance:** Inherits from `langgraph.prebuilt.chat_agent_executor.AgentState`.
*   **Fields:**
    *   `messages`: (Inherited) `List[BaseMessage]`, stores interaction history.
    *   `next`: `str`, the next worker to act.
    *   `user_prompt`: `str`, the initial user request.
    *   `df_ids`: `List[str]`, IDs of available DataFrames. **Default:** `[]`.
    *   `_config`: `Optional[RunnableConfig]`, LangGraph execution configuration. **Default:** `None`.
    *   `initial_description`: `Optional[InitialDescription]`, output of the initial analysis agent. **Default:** `None`.
    *   `cleaning_metadata`: `Optional[CleaningMetadata]`, output of the data cleaner agent. **Default:** `None`.
    *   `analysis_insights`: `Optional[AnalysisInsights]`, output of the main analyst agent. **Default:** `None`.
    *   `initial_analysis_agent`, `data_cleaner_agent`, `analyst_agent`: `Optional[BaseChatModel]`, can hold agent model instances. **Default:** `None`.
    *   `initial_analysis_complete`, `data_cleaning_complete`, `analyst_complete`, `file_writer_complete`, `visualization_complete`, `report_generator_complete`: `Optional[bool]`, flags for task completion. **Default:** `False`.
    *   `_count_`: `int`, internal step counter. **Default:** `0`.
    *   `_id_`: `str`, unique ID for the state instance. **Default:** `uuid.uuid4()`.
    *   `visualization_results`: `Optional[VisualizationResults]`, output of the visualization agent. **Default:** `None`.
    *   `report_results`: `Optional[ReportResults]`, output of the report generator agent. **Default:** `None`.
    *   `analysis_config`: `Optional[AnalysisConfig]`, user-defined analysis configurations. **Default:** `None`.

## 4. Agent System Components

### 4.1. Prompt Templates
These templates guide the LLMs for each agent.

#### `data_cleaner_prompt_template`
*   **Purpose:** Guides the Data Cleaner Agent to identify and fix data issues.
*   **Input Variables:** `dataset_description`, `data_sample`, `tool_descriptions`, `output_format`, `available_df_ids`.
*   **Key Instructions:** Role is "Data Cleaner Agent." Identifies issues (missing values, outliers, etc.), proposes and executes a step-by-step cleaning strategy using tools, explains reasoning, and outputs a summary in `CleaningMetadata` JSON schema.

#### `analyst_prompt_template_initial`
*   **Purpose:** Guides the Initial Analysis Agent (Data Describer and Sampler) to describe and sample the dataset.
*   **Input Variables:** `user_prompt`, `tool_descriptions`, `output_format`, `available_df_ids`.
*   **Key Instructions:** Role is "Data Describer and Sampler." Needs to provide a basic dataset description and data sample for the Data Cleaner Agent. Instructed to plan steps, use tools conservatively for description and sampling, and output in `InitialDescription` JSON schema. Warned against unnecessary tool calls and to report directly to the supervisor.

#### `analyst_prompt_template_main`
*   **Purpose:** Guides the main Analyst Agent for comprehensive EDA on cleaned data.
*   **Input Variables:** `cleaned_dataset_description`, `cleaning_metadata`, `tool_descriptions`, `output_format`, `available_df_ids`.
*   **Key Instructions:** Role is "Analyst Agent." Performs EDA (descriptive stats, correlations, anomalies), reasons step-by-step, recommends visualizations, and suggests next steps. Output in `AnalysisInsights` JSON schema.

#### `file_writer_prompt_template`
*   **Purpose:** Guides the File Writer Agent to write content to a file.
*   **Input Variables:** `file_name`, `content`, `file_type`, `tool_descriptions`, `available_df_ids`.
*   **Key Instructions:** Role is "agent that specializes in writing data." Task is to write provided `content` to `file_name` in `file_type`. Emphasizes writing content *only as requested* and leaving other tasks to other agents.

#### `visualization_prompt_template`
*   **Purpose:** Guides the Visualization Agent to create plots based on cleaned data and insights.
*   **Input Variables:** `cleaned_dataset_description`, `analysis_insights`, `tool_descriptions`, `output_format`, `available_df_ids`.
*   **Key Instructions:** Role is "Visualization Agent." Creates visualizations step-by-step using tools, explaining reasoning for each. Outputs a summary and visualization details in `VisualizationResults` JSON schema.

#### `report_generator_prompt_template`
*   **Purpose:** Guides the Report Generator Agent to compile a final report.
*   **Input Variables:** `cleaning_metadata`, `analysis_insights`, `visualization_results`, `tool_descriptions`, `output_format`, `available_df_ids`.
*   **Key Instructions:** Role is "Report Generator Agent." Generates a structured report combining text, stats, and visualizations. Explains report structure. Outputs summary and report path in `ReportResults` JSON schema.

### 4.2. Tool Definitions
Custom tools available to the agents. (Summarized - see previous detailed analysis for full parameter lists and return types if needed).

*   **`GetDataframeSchema` (func: `get_dataframe_schema`)**: Returns DataFrame schema and sample data. (Groups: `data_cleaning_tools`, `analyst_tools`, `file_writer_tools`, `visualization_tools`)
*   **`GetColumnNames` (func: `get_column_names`)**: Returns comma-separated column names. (Groups: `data_cleaning_tools`, `analyst_tools`, `visualization_tools`)
*   **`CheckMissingValues` (func: `check_missing_values`)**: Summarizes missing values. (Group: `data_cleaning_tools`)
*   **`DropColumn` (func: `drop_column`)**: Drops a column. (Group: `data_cleaning_tools`)
*   **`delete_rows`**: Deletes rows based on conditions. (Group: `data_cleaning_tools`)
*   **`FillMissingMedian` (func: `fill_missing_median`)**: Fills missing values with median. (Group: `data_cleaning_tools`)
*   **`QueryDataframe` (func: `query_dataframe`)**: Queries DataFrame with filters/operations. (Groups: `analyst_tools`, `data_cleaning_tools`)
*   **`GetData` (func: `get_data`)**: Retrieves flexible row/column/cell data. (Groups: `analyst_tools`, `data_cleaning_tools`, `visualization_tools`)
*   **`GetDescriptiveStatistics` (func: `get_descriptive_statistics`)**: Calculates descriptive stats. (Group: `analyst_tools`)
*   **`CalculateCorrelation` (func: `calculate_correlation`)**: Calculates Pearson correlation between two columns. (Group: `analyst_tools`)
*   **`PerformHypothesisTest` (func: `perform_hypothesis_test`)**: Performs a one-sample t-test. (Group: `analyst_tools`)
*   **`create_sample`**: Creates and saves an outline/sample points to a file. (Group: `analyst_tools`)
*   **`read_file`**: Reads specified lines from a file. (Groups: `file_writer_tools`, `report_generator_tools`)
*   **`write_file` (also as "WriteFile")**: Creates/saves a file with content. (Groups: `data_cleaning_tools`, `file_writer_tools`, `report_generator_tools`)
*   **`edit_file` (also as "EditFile")**: Inserts text at specific line numbers in a file. (Groups: `data_cleaning_tools`, `file_writer_tools`, `report_generator_tools`)
*   **`python_repl_tool` (also as "PythonREPL")**: Executes Python code with DataFrame access. (Groups: `analyst_tools`, `data_cleaning_tools`, `file_writer_tools`, `visualization_tools`, `report_generator_tools`)
*   **`create_histogram`**: Generates a histogram image (base64). (Group: `visualization_tools`)
*   **`create_scatter_plot`**: Generates a scatter plot image (base64). (Group: `visualization_tools`)
*   **`create_correlation_heatmap`**: Generates a correlation heatmap image (base64). (Group: `visualization_tools`)
*   **`create_box_plot`**: Generates a box plot image (base64). (Group: `visualization_tools`)
*   **`export_dataframe`**: Exports DataFrame to CSV, Excel, or JSON. (Groups: `analyst_tools`, `file_writer_tools`, `data_cleaning_tools`)
*   **`detect_and_remove_duplicates`**: Detects and removes duplicate rows. (Group: `data_cleaning_tools`)
*   **`convert_data_types`**: Converts column data types. (Group: `data_cleaning_tools`)
*   **`generate_html_report`**: Generates an HTML report from text/images. (Group: `report_generator_tools`)
*   **`calculate_correlation_matrix`**: Calculates and returns the full correlation matrix as JSON. (Group: `analyst_tools`)
*   **`detect_outliers`**: Detects outliers using IQR method. (Group: `analyst_tools`)
*   **`perform_normality_test`**: Performs Shapiro-Wilk normality test. (Group: `analyst_tools` - Note: Appended multiple times)
*   **`assess_data_quality`**: Provides a comprehensive data quality report (shape, missing values, types, duplicates, memory). (Groups: `analyst_tools`, `data_cleaning_tools`)
*   **`search_web_for_context`**: Performs web search via Tavily API. (Group: `analyst_tools`)
*   **`load_multiple_files`**: Loads multiple data files (CSVs, JSONs) into DataFrames. (Groups: `analyst_tools`, `data_cleaning_tools`)
*   **`merge_dataframes`**: Merges two DataFrames. (Groups: `analyst_tools`, `data_cleaning_tools`)
*   **`standardize_column_names`**: Standardizes column name formats (snake_case, lower, upper). (Group: `data_cleaning_tools`)
*   **`format_markdown_report`**: Formats a report into a Markdown file. (Group: `report_generator_tools`)
*   **`create_pdf_report`**: Converts an HTML file to a PDF report using `xhtml2pdf`. (Group: `report_generator_tools`)
*   **`train_ml_model`**: Trains Logistic/Linear Regression models. (Group: `analyst_tools`)
*   **`handle_categorical_encoding`**: Applies label or one-hot encoding to categorical columns. (Group: `data_cleaning_tools`)

### 4.3. Agent Creation Functions
These functions construct and configure the agents. All use `ChatOpenAI(model="gpt-4o-mini")`.

#### `create_data_cleaner_agent`
*   **Purpose:** Creates the Data Cleaner Agent.
*   **Parameters:** `initial_description: str`, `df_ids: List[str] = []`.
*   **Tools:** `data_cleaning_tools`.
*   **Prompt:** `data_cleaner_prompt_template` (formatted with `CleaningMetadata` schema).
*   **State Schema:** `State`. **Checkpointer/Store:** `MemorySaver()`, `in_memory_store`.
*   **Response Format:** `CleaningMetadata`. **Name/Version:** "data_cleaner", "v2".

#### `create_initial_analysis_agent`
*   **Purpose:** Creates the Initial Analysis Agent.
*   **Parameters:** `user_prompt: str`, `df_ids: List[str] = []`.
*   **Tools:** `analyst_tools`.
*   **Prompt:** `analyst_prompt_template_initial` (formatted with `InitialDescription` schema).
*   **State Schema:** `State`. **Checkpointer/Store:** `MemorySaver()`, `in_memory_store`.
*   **Response Format:** `InitialDescription`. **Name/Version:** "initial_analysis", "v2".

#### `create_analyst_agent`
*   **Purpose:** Creates the main Analyst Agent.
*   **Parameters:** `initial_description: str`, `df_ids: List[str] = []`.
*   **Tools:** `analyst_tools`.
*   **Prompt:** `analyst_prompt_template_main` (formatted with `AnalysisInsights` schema).
*   **State Schema:** `State`. **Checkpointer/Store:** `MemorySaver()`, `in_memory_store`.
*   **Response Format:** `AnalysisInsights`. **Name/Version:** "analyst", "v2".

#### `create_file_writer_agent`
*   **Purpose:** Creates the File Writer Agent.
*   **Parameters:** None.
*   **Tools:** `file_writer_tools`.
*   **Prompt:** No specific prompt template is explicitly formatted and passed into its `SystemMessage`; relies on ReAct default or general invocation.
*   **State Schema:** `State`. **Checkpointer/Store:** `MemorySaver()`, `in_memory_store`.
*   **Response Format:** Not specified. **Name/Version:** Not specified.

#### `create_visualization_agent`
*   **Purpose:** Creates the Visualization Agent.
*   **Parameters:** `df_ids: List[str] = []`.
*   **Tools:** `visualization_tools`.
*   **Prompt:** `visualization_prompt_template` (formatted with `VisualizationResults` schema).
*   **State Schema:** `State`. **Checkpointer/Store:** `MemorySaver()`, `in_memory_store`.
*   **Response Format:** `VisualizationResults`. **Name/Version:** "visualization", "v2".

#### `create_report_generator_agent`
*   **Purpose:** Creates the Report Generator Agent.
*   **Parameters:** `df_ids: List[str] = []`.
*   **Tools:** `report_generator_tools`.
*   **Prompt:** `report_generator_prompt_template` (formatted with `ReportResults` schema).
*   **State Schema:** `State`. **Checkpointer/Store:** `MemorySaver()`, `in_memory_store`.
*   **Response Format:** `ReportResults`. **Name/Version:** "report_generator", "v2".

### 4.4. Supervisor Node
The supervisor coordinates the agent team.

#### `make_supervisor_node` Function
*   **Purpose:** Factory to create the `supervisor_node` function.
*   **Parameters:** `llm` (BaseChatModel), `members` (list of worker agent names).
*   **Core Logic:** Defines routing `options` (agents + "FINISH"). Sets a `system_prompt` instructing the LLM to manage workers, decide the next worker based on user request and state, and respond "FINISH" when done. Defines a `Router` TypedDict for structured LLM output (`{"next": "agent_name_or_FINISH"}`).

#### `supervisor_node` (Inner Function)
*   **Purpose:** Orchestrates workflow by deciding the next agent or finishing.
*   **Input:** `state: State`.
*   **Core Logic:**
    *   Increments `_count_` in state.
    *   Dynamically updates its `system_prompt` with info on completed agents (e.g., "data_cleaner is complete").
    *   Invokes its LLM (with `Router` structured output) to get the next action (`goto`).
    *   If "FINISH", `goto` becomes `END`.
    *   Logs state and routing.
    *   Calls `update_memory`.
*   **Return Value:** `Command(goto=goto)`.

#### `update_memory` Function
*   **Purpose:** Stores the last message from the state into `InMemoryStore` under a `user_id` and new `memory_id` namespace. This allows for conversation history persistence, though its direct retrieval by the supervisor was noted as optional in the notebook code.
*   **Parameters:** `state`, `config`, `store`.

## 5. Graph Setup and Execution

*   **Initial Data Loading:** Sample data ("datafiniti/consumer-reviews-of-amazon-products") is loaded via `kagglehub`, registered into `global_df_registry`, and agents are instantiated with the `df_id`.
*   **Node Functions:** Agent-specific functions (e.g., `initial_analysis_node`) are defined to wrap agent calls, manage state, and format messages.
*   **Graph Definition (`StateGraph`):** A `StateGraph(State)` named `data_analysis_team_builder` is created. The supervisor (`coordinator_node`) and all agent node functions are added.
*   **Graph Edges:** The graph starts with the "supervisor". All agent nodes loop back to the "supervisor" upon completion.
*   **Graph Compilation:** The graph is compiled via `data_analysis_team_builder.compile()` using `MemorySaver` as `checkpointer` and `InMemoryStore` as `store`. The compiled graph is named `data_detective_graph`.
*   **Graph Visualization:** The graph structure is visualized using `draw_mermaid_png()`.
*   **Graph Execution (Streaming):**
    *   The primary execution method shown is `data_detective_graph.stream()`.
    *   Inputs include the initial `HumanMessage` (from `sample_prompt_text`), `user_prompt`, graph `config` (with `thread_id`, `user_id`, `recursion_limit`), and `df_ids`.
    *   `stream_mode="updates"` is used to get real-time state updates.
    *   Output chunks are printed and collected. State history is printed post-execution.
*   **Alternative Streaming Orchestration:** An `async` function `run_data_detective_stream` is provided as a reusable pattern for graph execution.
*   **Final State Access:** The script demonstrates accessing results (e.g., `analysis_insights`, `report_results`) from the graph's final state history.

## 6. Implemented Features Summary
The notebook implements a sophisticated multi-agent system capable of:
*   **Data Ingestion:** Loading datasets (initially from a Kaggle CSV, with tools for more general file loading).
*   **Automated Data Cleaning:** Identifying and handling issues like missing values, duplicates, and data type conversions through a dedicated Data Cleaner Agent and specialized tools.
*   **Exploratory Data Analysis (EDA):** Performing initial data description and sampling, followed by in-depth EDA including descriptive statistics, correlation analysis, anomaly detection, and normality testing via an Analyst Agent.
*   **Dynamic Visualization Generation:** Creating various plots (histograms, scatter plots, heatmaps, box plots) based on analysis findings through a Visualization Agent. Images are generated as base64 strings.
*   **Automated Report Generation:** Compiling findings from cleaning, EDA, and visualizations into structured reports in HTML, Markdown, and PDF formats via a Report Generator Agent.
*   **Web Search for Context:** Utilizing the Tavily API via a tool to fetch external information that might be relevant for analysis.
*   **Basic Machine Learning:** Training simple classification (Logistic Regression) and regression (Linear Regression) models, including data splitting and metric reporting (accuracy, RMSE).
*   **Workflow Orchestration:** Using a supervisor agent to manage the sequence of operations and delegate tasks to specialized agents.
*   **State Management & Checkpointing:** Employing LangGraph's state management with `MemorySaver` for persisting graph state.

## 7. Unimplemented Features (Based on Tech Spec)
Direct comparison with the `Project_Tech_Spec_Intelligent_Data_Detective.pdf` was not possible due to limitations in parsing the PDF content with the available tools. This section would typically list features from the spec not yet implemented or fully realized in the notebook. Users familiar with the tech spec are encouraged to contribute to this section by comparing the documented features above with the specification.

Potential areas often detailed in tech specs that might not be fully implemented in a notebook context include:
*   **User Interface (UI):** The notebook is script-based. A dedicated UI (e.g., Gradio, Streamlit) for user interaction is typically a separate component.
*   **Advanced RAG Techniques:** While a web search tool exists, more sophisticated RAG (Retrieval Augmented Generation) involving vector databases, document chunking for domain-specific knowledge, or dedicated RAG agents might be specified.
*   **Deployment:** The notebook provides the logic but is not a deployed application. Deployment strategies (cloud, local server) would be outlined in a tech spec.
*   **Security & Data Privacy:** Beyond basic API key handling, a deployed application would require more robust security and data privacy measures.
*   **Detailed Roadmap Features:** Specific sub-features under Roadmap phases (especially Phase 2 and Future) like interactive dashboard elements, advanced anomaly detection algorithms, or user feedback loops for model retraining would be areas for comparison.
*   **Specific Agent Capabilities from "Detailed Agent Logic":** The tech spec might outline more nuanced conversational abilities, error handling, or specific tool interaction patterns for agents that go beyond the current implementation.

This compiled document should serve as a good reference for the notebook's current capabilities and structure.
