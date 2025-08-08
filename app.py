import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time

# Import your existing modules
from src.common.database_pool import get_pooled_db_connection
from src.pipelines.plan_backtest_v2.simulation_runner import ChronologicalSimulationRunner
from src.pipelines.plan_backtest_v2.plan import Plan
from restricted_config import (
    RESTRICTED_DATA, REAL_DATA_PASSWORD, 
    ALLOWED_GROUP_IDS, GROUP_ID_MAPPING, 
    get_restricted_member_name
)

# Configure the page
st.set_page_config(
    page_title="Plan Comparison Simulation Tool",
    page_icon="📊",
    layout="wide"
)

try:
    logo_path = r'src\pipelines\plan_backtest_v2\misc\ua-logo.png'
    st.image(logo_path, width=400)  # Adjust width as needed
except Exception as e:
    st.error(f"Error loading logo: {str(e)}")

st.warning("""
⚠️ **BETA TEST** This tool is still in development. 
Results are preliminary and may not be 100% accurate.
""")

# Set up session state for admin mode and stop button
if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False

if 'stop_simulation' not in st.session_state:
    st.session_state.stop_simulation = False

# Title and description
st.title("Plan Comparison Simulation Tool")
st.caption("Analyze and compare different plan configurations")

# Function to check password
def check_password(password):
    """Check if the password is correct for admin mode"""
    return password == REAL_DATA_PASSWORD

# Function to fetch available plans for a selected year
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_available_plans(year):
    """Fetch available plans for the selected year from the database"""
    try:
        # Connect to the database
        engine = get_pooled_db_connection('Datamart')
        
        # Use your SQL query with the selected year
        query = f"""
        WITH PlanPopularity AS (
            SELECT 
                dp.PlanID,
                dp.PlanShortName as PlanName,
                COUNT(DISTINCT dg.GroupID) AS GroupCount,
                RANK() OVER (ORDER BY COUNT(DISTINCT dg.GroupID) DESC) AS PopularityRank
            FROM Datamart.dbo.FactGroupPlan fgp
            JOIN Datamart.dbo.DimPlan dp ON fgp.PlanKey = dp.PlanKey
            JOIN Datamart.dbo.DimPlanPackage dpp ON fgp.PlanPackageKey = dpp.PlanPackageKey
            JOIN Datamart.dbo.DimDate dd ON fgp.DateKey = dd.DateKey
            JOIN Datamart.dbo.DimGroup dg ON fgp.GroupKey = dg.GroupKey
            LEFT JOIN Repository.ref.WLTPlanMatrixCrosswalk cr ON cr.INNPlanID = dp.PlanID
            LEFT JOIN Repository.ref.PlanMatrix med ON cr.PlanCode = med.PlanCode AND med.Plan_Year = {year}
            WHERE 
                dp.PlanType IN ('Medical')
                AND dd.Year = {year}
                AND dpp.ActiveFlag = 'Y'
            GROUP BY 
                dp.PlanID, 
                dp.PlanShortName
        )
        SELECT 
            PlanID,
            PlanName,
            GroupCount,
            PopularityRank
        FROM PlanPopularity
        ORDER BY 
            PopularityRank,
            PlanID
        """
        
        # Execute query and fetch results
        plan_df = pd.read_sql(query, engine)
        
        # Convert to dictionary for easy lookup
        plan_dict = dict(zip(plan_df['PlanID'], plan_df['PlanName']))
        
        return plan_dict
        
    except Exception as e:
        st.error(f"Error fetching plans: {str(e)}")
        return {}  # Return empty dict if there's an error

# Function to fetch available groups
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_available_groups(year):
    """Fetch available groups that have claims in the specified year"""
    try:
        # Connect to the database
        engine = get_pooled_db_connection('Datamart')
        
        # Query to get groups with claims in the specified year
        query = f"""
        SELECT DISTINCT 
            dc.GroupID,
            dg.GroupName
        FROM Datamart.dbo.DimClaim dc
        JOIN Datamart.dbo.DimGroup dg ON dc.GroupID = dg.GroupID
        WHERE dc.ServiceFromDate LIKE '{year}%'
        AND dc.ClaimStatus IN ('Adjudicated', 'Paid')
        and dc.ClaimType not in ('Dental', 'Vision', 'Drug') 
        AND dc.IsCurrent = 'Y'
        AND dc.GroupID between 120 and 79999
        """
        
        # If obfuscation is enabled and not in admin mode, filter to only allowed groups
        if RESTRICTED_DATA and not st.session_state.admin_mode:
            query += f" AND dc.GroupID IN ({','.join(map(str, ALLOWED_GROUP_IDS))})"
        
        query += " ORDER BY dc.GroupID"
        
        # Execute query and fetch results
        group_df = pd.read_sql(query, engine)
        
        # Convert to dictionary for easy lookup
        if RESTRICTED_DATA and not st.session_state.admin_mode:
            # Use restricted names for the allowed groups
            group_dict = {row['GroupID']: GROUP_ID_MAPPING.get(row['GroupID'], f"Test Group {row['GroupID']}") 
                         for _, row in group_df.iterrows()}
        else:
            # Use real names
            group_dict = dict(zip(group_df['GroupID'], group_df['GroupName']))
        
        return group_dict
        
    except Exception as e:
        st.error(f"Error fetching groups: {str(e)}")
        # Return sample data if there's an error
        if RESTRICTED_DATA and not st.session_state.admin_mode:
            return {15702: "Test Group Alpha"}
        else:
            return {15702: "Sample Group"}

# Function to fetch members for a specific group
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_members_by_group(group_id, year):
    """Fetch members belonging to a specific group with claims in the specified year"""
    try:
        # Connect to the database
        engine = get_pooled_db_connection('Datamart')
        
        # Query to get members with claims for the specified group and year
        query = f"""
        SELECT DISTINCT 
            dc.MemberID,
            m.EmployeeID,
            m.FirstName + ' ' + m.LastName AS MemberName
        FROM Datamart.dbo.DimClaim dc
        JOIN Datamart.dbo.DimMember m ON dc.MemberID = m.MemberID
        WHERE dc.GroupID = {group_id}
        AND dc.ServiceFromDate LIKE '{year}%'
        AND dc.ClaimStatus IN ('Adjudicated', 'Paid')
        and dc.ClaimType not in ('Dental', 'Vision', 'Drug') 
        AND dc.IsCurrent = 'Y'
        ORDER BY m.EmployeeID, dc.MemberID
        """
        
        # Execute query and fetch results
        member_df = pd.read_sql(query, engine)
        
        # Convert to dictionary for easy lookup
        if RESTRICTED_DATA and not st.session_state.admin_mode:
            member_dict = {row['MemberID']: get_restricted_member_name(row['MemberID'], row['MemberName']) 
                          for _, row in member_df.iterrows()}
        else:
            # Use real names
            member_dict = dict(zip(member_df['MemberID'], member_df['MemberName']))
        
        return member_dict
        
    except Exception as e:
        st.error(f"Error fetching members: {str(e)}")
        # Return sample data if there's an error
        if RESTRICTED_DATA and not st.session_state.admin_mode:
            return {
                f'{group_id}-106-1': "Employee 106",
                f'{group_id}-29-1': "Employee 29"
            }
        else:
            return {
                f'{group_id}-106-1': "John Smith",
                f'{group_id}-29-1': "Jane Doe"
            }

# Function to fetch the existing plan for a given group and year
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_existing_plan_for_group(group_id, year):
    """Fetch the existing plan ID for a specific group and year"""
    try:
        # Connect to the database
        engine = get_pooled_db_connection('Datamart')
        
        # Query to get the current plan for the group in the specified year
        query = f"""
        SELECT TOP 1
            dp.PlanID,
            dp.PlanShortName as PlanName
        FROM Datamart.dbo.FactGroupPlan fgp
        JOIN Datamart.dbo.DimPlan dp ON fgp.PlanKey = dp.PlanKey
        JOIN Datamart.dbo.DimGroup dg ON fgp.GroupKey = dg.GroupKey
        JOIN Datamart.dbo.DimDate dd ON fgp.DateKey = dd.DateKey
        WHERE dg.GroupID = {group_id}
        AND dd.Year = {year}
        AND dp.PlanType = 'Medical'
        ORDER BY fgp.DateKey DESC
        """
        
        # Execute query and fetch result
        result = pd.read_sql(query, engine)
        
        if not result.empty:
            return result.iloc[0]['PlanID'], result.iloc[0]['PlanName']
        else:
            return None, None
        
    except Exception as e:
        st.error(f"Error fetching existing plan: {str(e)}")
        return None, None  # Return None if there's an error

# Helper function to format currency values safely
def format_currency(df, columns):
    """Format numeric columns as currency without modifying original DataFrame dtype"""
    formatted_df = df.copy()
    for col in columns:
        if col in formatted_df.columns:
            # Create a new string column with formatted values
            formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
    return formatted_df

# Helper function to format percentages safely
def format_percentage(df, columns):
    """Format numeric columns as percentages without modifying original DataFrame dtype"""
    formatted_df = df.copy()
    for col in columns:
        if col in formatted_df.columns:
            # Create a new string column with formatted values
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
    return formatted_df

def green_info_block(text, markdown=False):
    """
    Creates a green info block with customizable styling.
    
    Parameters:
    - text: The text content to display in the info block
    - markdown: Boolean, whether to process markdown formatting (default: True)
    """
    import re
    
    if markdown:
        # Convert markdown syntax to HTML
        # Bold text
        processed_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Italic text
        processed_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', processed_text)
        
        # Convert markdown bullet points to HTML list items
        processed_text = re.sub(r'- (.*?)(?=\n|$)', r'• \1', processed_text)
        
        # Ensure proper line breaks
        processed_text = processed_text.replace('\n', '<br>')
    else:
        # For simple text, just handle line breaks
        processed_text = text.replace('\n', '<br>')
    
    # Create the styled div
    st.markdown(
        f"""
        <div style="background-color: #e6f7e6; 
                    color: #0e6928; 
                    padding: 16px; 
                    border-radius: 8px; 
                    border-left: 6px solid #28a745;
                    margin-bottom: 20px;">
            {processed_text}
        </div>
        """, 
        unsafe_allow_html=True
    )

# Sidebar elements
with st.sidebar:
    # Display the current data mode
    green_info_block(f"Current mode: {'Admin' if st.session_state.admin_mode else 'Restricted'}")
    
    with st.expander("Settings", expanded=False):
        # Admin mode sidebar section
        if not st.session_state.admin_mode:
            password = st.text_input("Enter admin password to see real data:", type="password")
            if st.button("Login"):
                if check_password(password):
                    st.session_state.admin_mode = True
                    st.success("Admin mode enabled!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Incorrect password!")
        else:
            st.success("Admin mode enabled")
            if st.button("Logout"):
                st.session_state.admin_mode = False
                st.cache_data.clear()
                st.rerun()
    
    # Now add testing parameters section below the button
    st.markdown("<hr/>", unsafe_allow_html=True)  # Optional: add a divider
    
    # Year selection
    st.subheader("Select Year")
    year = st.selectbox("Year", [2024, 2023, 2022], index=0)
    
    # Group selection
    st.subheader("Select Group")
    groups = get_available_groups(year)
    
    # Format the display options to show ID - Name
    group_options = [f"{gid} - {gname}" for gid, gname in groups.items()]
    group_id_map = {f"{gid} - {gname}": gid for gid, gname in groups.items()}
    
    # Use built-in search with the selectbox
    selected_group_option = st.selectbox(
        "Group (type to search)",
        options=group_options,
        index=2 if group_options else None
    )
    
    # Map selected option back to group ID
    group_id = group_id_map.get(selected_group_option) if selected_group_option else None
    
    # Month range selection
    st.subheader("Date Range")
    use_month_range = st.checkbox("Filter by Month Range", value=False)
    if use_month_range:
        start_month, end_month = st.select_slider(
            "Select Month Range:", 
            options=list(range(1, 13)),
            value=(1, 12),
            format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1]
        )
    else:
        start_month, end_month = 1, 12
    
    # Plans selection
    st.subheader("Select Plans")
    plans = get_available_plans(year)
    
    # Get the current plan for this group
    current_plan_id, current_plan_name = get_existing_plan_for_group(group_id, year) if group_id else (None, None)
    if current_plan_id:
        green_info_block(f"Current plan: {current_plan_id} - {current_plan_name}")
    
    # Add current plan at the top of the list if available
    if current_plan_id and current_plan_id in plans:
        plan_options = [f"{current_plan_id} - {plans[current_plan_id]}"] + [
            f"{p} - {plans[p]}" for p in plans.keys() if p != current_plan_id
        ]
    else:
        plan_options = [f"{p} - {plans[p]}" for p in plans.keys()]
    
    plan_id_map = {f"{pid} - {pname}": pid for pid, pname in plans.items()}
    
    # Allow selecting multiple plans
    selected_plan_options = st.multiselect(
        "Plans to Compare (type to search):",
        options=plan_options,
        default=[f"{current_plan_id} - {plans[current_plan_id]}"] if current_plan_id and current_plan_id in plans else []
    )
    
    # Map selected options back to plan IDs
    selected_plans = [plan_id_map.get(option) for option in selected_plan_options if option in plan_id_map]
    
    # Ready to run message
    run_disabled = len(selected_plans) == 0 or group_id is None
    
    if run_disabled:
        st.warning("Please select group and at least one plan")
    else:
        # We'll update this message based on testing parameters later
        ready_message_container = st.empty()
    
    # Run simulation button
    run_button = st.button("Run Simulation", disabled=run_disabled, type="primary")
    
    # Now add testing parameters section below the button
    st.markdown("<hr/>", unsafe_allow_html=True)  # Optional: add a divider
    
    # Testing parameters section
    st.markdown(
        "<div style='text-align: center; background-color: #fff3cd; padding: 10px; border-radius: 5px;'>"
        "⚠️ <b>Parameters below are for testing and QA only</b> ⚠️"
        "</div>", 
        unsafe_allow_html=True
    )
    
    st.markdown("<div style='margin: 10px 0px;'></div>", unsafe_allow_html=True)

    with st.expander("Processing Options", expanded=False):
        # Workers and batch size
        num_workers_per_plan = st.slider("Number of Workers (Per Plan):", min_value=1, max_value=5, value=4, 
                            help="More workers can speed up processing but use more CPU")
        num_workers = num_workers_per_plan * len(selected_plans)
        
        batch_size = st.slider("Batch Size:", min_value=1, max_value=50, value=30,
                            help="Number of families processed together in each batch")
        
        # Add parallelization strategy selection when multiple plans are selected
        if len(selected_plans) > 1:
            st.subheader("Parallelization Strategy")
            parallelization_strategy = st.radio(
                "Choose parallelization strategy:",
                options=[
                    "Fully PARALLEL",
                    "PARALLEL plans with SEQUENTIAL claims",
                    "SEQUENTIAL plans with PARALLEL claims", 
                    "Fully SEQUENTIAL"
                ],
                index=2,  # Default to fully parallel
                help="Different strategies work better for different scenarios"
            )
            
            # Convert selection to strategy flags
            if parallelization_strategy.startswith("Fully PARALLEL"):
                st.session_state.parallel_claims = True
                st.session_state.parallel_plans = True
            elif parallelization_strategy.startswith("PARALLEL plans with SEQUENTIAL claims"):
                st.session_state.parallel_claims = False
                st.session_state.parallel_plans = True
            elif parallelization_strategy.startswith("SEQUENTIAL plans with PARALLEL claims"):
                st.session_state.parallel_claims = True
                st.session_state.parallel_plans = False
            else:  # Fully sequential
                st.session_state.parallel_claims = False
                st.session_state.parallel_plans = False
        else:
            # If only one plan, we only have parallel or sequential claims
            single_plan_parallelization = st.radio(
                "Claims processing:",
                options=["Parallel claims processing", "Sequential claims processing"],
                index=0,  # Default to parallel claims
                help="Parallel is faster but sequential may be more reliable for some plans"
            )
            st.session_state.parallel_claims = single_plan_parallelization.startswith("Parallel")
            st.session_state.parallel_plans = False  # Not applicable for single plan
    
    with st.expander("Testing Options", expanded=False):
        # Member selection
        st.subheader("Member Options")
        use_specific_members = st.checkbox("Use Specific Members", value=False)
        
        selected_members = None
        if use_specific_members and group_id:
            members = get_members_by_group(group_id, year)
            member_options = [f"{mid} - {mname}" for mid, mname in members.items()]
            member_id_map = {f"{mid} - {mname}": mid for mid, mname in members.items()}
            
            selected_member_options = st.multiselect(
                "Select Members:",
                options=member_options
            )
            
            # Map selected options back to member IDs
            if selected_member_options:
                selected_members = [member_id_map.get(option) for option in selected_member_options if option in member_id_map]
        
        # Detailed member analysis option
        detailed_member_analysis = st.checkbox("Enable Member Analysis", value=False,
                                            help="Analyze top members in detail")
        
    # Now update the ready message based on testing parameters
    if not run_disabled:
        members_message = f"{len(selected_members) if selected_members else 'all'} members"
        ready_message_container.info(f"Ready to run simulation for {len(selected_plans)} plans and {members_message}")

# Main layout
def main():
    # Create tabs for main sections
    setup_tab, results_tab = st.tabs(["Simulation Setup", "Simulation Results"])
    
    # First tab - additional setup options (if needed)
    with setup_tab:
        st.header("Simulation Configuration")
        
        # Status message showing selected parameters
        col1, col2 = st.columns(2)
        
        with col1:
            if group_id:
                group_name = groups.get(group_id, "Unknown Group")
                green_info_block(f"Selected Group: <b>{group_id} - {group_name}<b>")
            else:
                st.warning("No group selected. Please select a group in the sidebar.")
        
        with col2:
            if selected_plans:
                plan_names = [f"{p} ({plans.get(p, 'Unknown')})" for p in selected_plans]
                green_info_block(f"Selected Plans: <b>{', '.join(plan_names)}<b>")
            else:
                st.warning("No plans selected. Please select at least one plan in the sidebar.")
        
        # Month range display
        if use_month_range:
            month_names = ["January", "February", "March", "April", "May", "June", 
                          "July", "August", "September", "October", "November", "December"]
            green_info_block(f"Analyzing months: <b>{month_names[start_month-1]} to {month_names[end_month-1]}<b>")
                
        # Member selection details
        if use_specific_members and selected_members:
            st.subheader("Selected Members")
            
            members_dict = get_members_by_group(group_id, year)
            member_data = []
            
            for member_id in selected_members:
                member_name = members_dict.get(member_id, f"Unknown ({member_id})")
                member_data.append({"Member ID": member_id, "Name": member_name})
            
            if member_data:
                member_df = pd.DataFrame(member_data)
                member_df.index = range(1, len(member_df) + 1)
                st.dataframe(member_df, use_container_width=True)
        
        # Help text with run instructions
        st.markdown("---")
        st.markdown("""
        ### Instructions
        1. **Select parameters** in the sidebar: year, group, plans, and any optional filters
        2. **Click "Run Simulation"** to start the analysis
        3. Results will appear in the "Simulation Results" tab
        """)
    
    # Results Tab - only show content if we have results
    with results_tab:
        if run_button or 'simulation_results' in st.session_state:
            # Run the simulation if button was clicked
            if run_button:
                # Reset stop flag at the beginning of a new run
                st.session_state.stop_simulation = False
                
                # Store parameters in session state
                st.session_state.simulation_params = {
                    'year': year,
                    'group_id': group_id,
                    'group_name': groups.get(group_id, "Unknown Group"),
                    'selected_plans': selected_plans,
                    'plan_names': {p: plans.get(p, f"Plan {p}") for p in selected_plans},
                    'month_range': (start_month, end_month) if use_month_range else None,
                    'selected_members': selected_members,
                    'num_workers': num_workers,
                    'batch_size': batch_size,
                    'detailed_member_analysis': detailed_member_analysis,
                    'current_plan_id': current_plan_id,
                    'current_plan_name': current_plan_name,
                    'parallel_claims': getattr(st.session_state, 'parallel_claims', True),
                    'parallel_plans': getattr(st.session_state, 'parallel_plans', len(selected_plans) > 1)
                }
                
                # Create a progress indicator and timer
                progress_placeholder = st.empty()
                timer_placeholder = st.empty()
                
                # Add stop button (will be removed after completion)
                stop_button_placeholder = st.empty()
                stop_clicked = stop_button_placeholder.button("🔴 Stop Simulation")
                if stop_clicked:
                    st.session_state.stop_simulation = True
                    st.warning("Stopping simulation... Please wait.")
                
                # Start time for the processing
                processing_start_time = time.time()
                st.session_state.start_time = processing_start_time
                
                with st.spinner("Running simulation..."):
                    # Show initial progress message
                    progress_placeholder.info(f"Starting simulations for {len(selected_plans)} plans...")
                    
                    # Update timer on initial start
                    timer_placeholder.info(f"⏱️ Elapsed time: 0.00 seconds")
                    
                    # Run the simulation
                    simulation_results = run_simulation(
                        st.session_state.simulation_params,
                        progress_callback={
                            'progress_placeholder': progress_placeholder,
                            'timer_placeholder': timer_placeholder
                        }
                    )
                    
                    st.session_state.simulation_results = simulation_results
                    
                    # Remove stop button after completion
                    stop_button_placeholder.empty()
                    
                    # Clear progress elements
                    progress_placeholder.empty()
                    timer_placeholder.empty()
            
            # Display simulation results
            display_simulation_results(
                st.session_state.get('simulation_results', {}),
                st.session_state.get('simulation_params', {})
            )
            
            # Display completion message at bottom of page
            if 'simulation_params' in st.session_state:
                st.success(f"✅ Completed in {time.time() - st.session_state.get('start_time', time.time()):.2f} seconds")
        else:
            green_info_block("Run a simulation to view results here.")

def run_simulation(params, progress_callback=None):
    """Run the simulation with the given parameters"""
    try:
        # Create simulation runner
        runner = ChronologicalSimulationRunner()
        
        # Extract progress callback elements if provided
        progress_placeholder = progress_callback.get('progress_placeholder') if progress_callback else None
        timer_placeholder = progress_callback.get('timer_placeholder') if progress_callback else None
        
        # Determine whether to use single-plan or multi-plan approach based on strategy
        if len(params['selected_plans']) == 1:
            # Single plan - always use parallel claims if requested
            plan_id = params['selected_plans'][0]
            
            # Update progress
            if progress_placeholder:
                progress_placeholder.info(f"Processing plan: {plan_id}")
            
            # Run the simulation
            if params.get('parallel_claims', True):
                # Use parallel processing for a single plan
                group_results, member_results = runner.run_parallel_group_simulation(
                    plan_id=plan_id,
                    year=params['year'],
                    group_id=params['group_id'],
                    max_workers=params['num_workers'],
                    test_member_ids=params['selected_members'],
                    month_range=params['month_range'],
                    batch_size=params['batch_size'],
                    show_progress=False
                )
            else:
                # Use sequential processing for a single plan
                group_results, member_results = runner.run_group_simulation(
                    plan_id=plan_id,
                    year=params['year'],
                    group_id=params['group_id'],
                    test_member_ids=params['selected_members'],
                    month_range=params['month_range'],
                )
            
            # Store results in the expected format
            results = {
                plan_id: {
                    'group_results': group_results,
                    'member_results': member_results
                }
            }
            
        else:
            # Multiple plans - use strategy-specific approach
            parallel_plans = params.get('parallel_plans', True)
            parallel_claims = params.get('parallel_claims', True)
            
            if progress_placeholder:
                strategy_name = "unknown"
                if parallel_plans and parallel_claims:
                    strategy_name = "fully parallel"
                elif parallel_plans and not parallel_claims:
                    strategy_name = "parallel plan processing and sequential claims processing"
                elif not parallel_plans and parallel_claims:
                    strategy_name = "sequential plan processing with parallel claims processing"
                else:
                    strategy_name = "fully sequential"
                    
                progress_placeholder.info(f"Processing {len(params['selected_plans'])} plans using {strategy_name} strategy")
            
            if parallel_plans and parallel_claims:
                # Fully parallel - both plans and claims are processed in parallel
                results = runner.run_parallel_multi_plan_simulation(
                    plan_ids=params['selected_plans'],
                    year=params['year'],
                    group_id=params['group_id'],
                    max_workers=params['num_workers'],
                    test_member_ids=params['selected_members'],
                    month_range=params['month_range'],
                    batch_size=params['batch_size'],
                    show_progress=False
                )
            elif parallel_plans and not parallel_claims:
                # Hybrid: Parallel plans with sequential claims
                results = runner.run_parallel_plans_sequential_claims(
                    plan_ids=params['selected_plans'],
                    year=params['year'],
                    group_id=params['group_id'],
                    max_workers=params['num_workers'],
                    test_member_ids=params['selected_members'],
                    month_range=params['month_range'],
                    show_progress=False
                )
            elif not parallel_plans and parallel_claims:
                # Hybrid: Sequential plans with parallel claims
                results = runner.run_sequential_plans_parallel_claims(
                    plan_ids=params['selected_plans'],
                    year=params['year'],
                    group_id=params['group_id'],
                    max_workers=params['num_workers'],
                    test_member_ids=params['selected_members'],
                    month_range=params['month_range'],
                    batch_size=params['batch_size'],
                    show_progress=False
                )
            else:
                # Fully sequential - no parallelization
                results = {}
                for plan_id in params['selected_plans']:
                    if progress_placeholder:
                        progress_placeholder.info(f"Processing plan {plan_id} sequentially")
                    
                    group_results, member_results = runner.run_group_simulation(
                        plan_id=plan_id,
                        year=params['year'],
                        group_id=params['group_id'],
                        test_member_ids=params['selected_members'],
                        month_range=params['month_range']
                    )
                    
                    results[plan_id] = {
                        'group_results': group_results,
                        'member_results': member_results
                    }
                    
                    if timer_placeholder and 'start_time' in st.session_state:
                        elapsed = time.time() - st.session_state.start_time
                        timer_placeholder.info(f"⏱️ Elapsed time: {elapsed:.2f} seconds")
        
        # Extract and prepare summary data from results
        summary_data = []
        for plan_id, plan_results in results.items():
            if 'group_results' in plan_results and plan_results['group_results']:
                group_res = plan_results['group_results']
                plan_name = params['plan_names'].get(plan_id, f"Plan {plan_id}")
                
                # Calculate metrics
                member_total = group_res.get("total_member_responsibility", 0)
                plan_total = group_res.get("total_plan_responsibility", 0)
                total_cost = member_total + plan_total
                member_count = group_res.get("member_count", 0)
                avg_cost = member_total / member_count if member_count > 0 else 0
                
                summary_data.append({
                    "Plan ID": plan_id,
                    "Plan Name": plan_name,
                    "Member Responsibility": member_total,
                    "Plan Responsibility": plan_total,
                    "Total Cost": total_cost,
                    "Avg Cost Per Member": avg_cost,
                    "Is Current Plan": plan_id == params['current_plan_id']
                })
        
        # Create summary DataFrame
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values("Is Current Plan")
        else:
            summary_df = pd.DataFrame()
        
        # Process coverage code data if available
        coverage_data = {}
        for plan_id, plan_results in results.items():
            if ('group_results' in plan_results and plan_results['group_results'] and 
                'coverage_summary' in plan_results['group_results']):
                
                coverage_summary = plan_results['group_results']['coverage_summary']
                if not coverage_summary.empty:
                    coverage_data[plan_id] = coverage_summary
        
        # Process diagnosis grouper data if available
        diagnosis_data = {}
        for plan_id, plan_results in results.items():
            if ('group_results' in plan_results and plan_results['group_results'] and 
                'diagnosis_summary' in plan_results['group_results']):
                
                diagnosis_summary = plan_results['group_results']['diagnosis_summary']
                if not diagnosis_summary.empty:
                    diagnosis_data[plan_id] = diagnosis_summary
        
        # Process top member data if requested
        top_member_data = {}
        if params['detailed_member_analysis']:
            for plan_id, plan_results in results.items():
                if 'member_results' in plan_results and plan_results['member_results']:
                    # Calculate cost for each member
                    member_costs = []
                    for member_id, data in plan_results['member_results'].items():
                        member_costs.append({
                            "Member ID": member_id,
                            "Member Responsibility": data.get("member_total", 0),
                            "Plan Responsibility": data.get("plan_total", 0),
                            "Total Cost": data.get("member_total", 0) + data.get("plan_total", 0)
                        })
                    
                    # Convert to DataFrame, sort by cost, and take top 10
                    if member_costs:
                        member_df = pd.DataFrame(member_costs)
                        member_df = member_df.sort_values("Total Cost", ascending=False).head(20)
                        top_member_data[plan_id] = member_df
        
        return {
            'summary': summary_df,
            'raw_results': results,
            'coverage_data': coverage_data,
            'diagnosis_data': diagnosis_data,
            'top_member_data': top_member_data
        }
    
    except Exception as e:
        st.error(f"Error running simulation: {str(e)}")
        # Return empty result structure
        return {
            'summary': pd.DataFrame(),
            'raw_results': {},
            'coverage_data': {},
            'diagnosis_data': {},
            'top_member_data': {}
        }

def display_simulation_results(results, params):
    """Display the simulation results"""
    summary_df = results.get('summary', pd.DataFrame())
    
    if summary_df.empty:
        st.error("No results available to display.")
        return
    
    # Warning for specific member selection
    if params.get('selected_members'):
        st.warning(f"⚠️ TEST MODE: Analyzing {len(params['selected_members'])} selected members from Group {params.get('group_id', 'Unknown')}")
    
    # Main Results Header
    st.header(f"Simulation Results for {params.get('group_name', 'Unknown Group')}")
    st.caption(f"Year: {params.get('year', 'Unknown')}" + 
              (f", Months: {params['month_range'][0]}-{params['month_range'][1]}" 
               if params.get('month_range') else ""))
               
    # Success message - moved to top of results
    st.success("All plans processed successfully")
    
    # Summary Statistics
    st.subheader("Plan Comparison Summary")
    
    # Add member and claim count information
    # Fetch this from the first plan's results
    first_plan_id = list(results['raw_results'].keys())[0]
    first_plan_results = results['raw_results'][first_plan_id]['group_results']
    green_info_block(f"Analysis includes {first_plan_results.get('member_count', 0)} members across {first_plan_results.get('claim_count', 0)} claims")
    
    # Reorder summary to put current plan first
    current_plan_rows = summary_df[summary_df['Is Current Plan']]
    other_plan_rows = summary_df[~summary_df['Is Current Plan']]
    summary_df = pd.concat([current_plan_rows, other_plan_rows]).reset_index(drop=True)
    summary_df.index = range(1, len(summary_df) + 1)

    # Summary Table - Use safe formatting function
    formatted_df = format_currency(summary_df, 
        ['Member Responsibility', 'Plan Responsibility', 'Total Cost', 'Avg Cost Per Member'])

    st.dataframe(
        formatted_df,  # Remove .style
        column_config={
            "Is Current Plan": st.column_config.CheckboxColumn(
                "Current Plan",
                help="Whether this is the group's current plan"
            )
        },
        use_container_width=True
    )
    
    # Total Cost Comparison Bar Chart
    st.subheader("Total Cost Comparison")
    
    # Reorder DataFrame to put current plan first
    current_plan_rows = summary_df[summary_df['Is Current Plan']]
    other_plan_rows = summary_df[~summary_df['Is Current Plan']]
    sorted_summary_df = pd.concat([current_plan_rows, other_plan_rows])
    
    fig = go.Figure()
    
    # Track whether legend entries have been added
    member_legend_added = False
    plan_legend_added = False
    
    for i, row in sorted_summary_df.iterrows():
        # Determine if this is the current plan
        is_current = row['Is Current Plan']
        
        # Create plan name with Plan ID and full name, and checkmark for current plan
        plan_display_name = f"{row['Plan ID']} - {params['plan_names'].get(row['Plan ID'], 'Unknown Plan')}" + (" ✓" if is_current else "")
        
        # Get costs
        member_cost = row['Member Responsibility']
        plan_cost = row['Plan Responsibility']
        
        # Add member responsibility - Green color to match visualization.py
        fig.add_trace(go.Bar(
            name='Member Responsibility' if not member_legend_added else '',
            x=[plan_display_name],
            y=[member_cost],
            marker_color='#2E8B57',  # Dark green from visualization.py
            text=[f"${member_cost:,.2f}"],
            textposition='auto',
            textfont=dict(color='white'),  # White text for better readability
            offsetgroup=i,
            showlegend=not member_legend_added
        ))
        member_legend_added = True
        
        # Add plan responsibility - Purple color to match visualization.py
        fig.add_trace(go.Bar(
            name='Plan Responsibility' if not plan_legend_added else '',
            x=[plan_display_name],
            y=[plan_cost],
            marker_color='#8A2BE2',  # Purple from visualization.py
            text=[f"${plan_cost:,.2f}"],
            textposition='auto',
            textfont=dict(color='white'),  # White text for better readability
            offsetgroup=i,
            showlegend=not plan_legend_added
        ))
        plan_legend_added = True

    # Update layout
    fig.update_layout(
        barmode='stack',
        title='Cost Breakdown by Plan',
        xaxis_title='Plan (ID - Name)',
        yaxis_title='Cost ($)',
        legend_title='Cost Type',
        height=500,
        # Remove gridlines
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        # Add legend at the top
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # Set margin
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True, key="main_cost_comparison")
    
    # Reorder plan IDs to put current plan first
    current_plan_ids = summary_df[summary_df['Is Current Plan']]['Plan ID'].tolist()
    other_plan_ids = summary_df[~summary_df['Is Current Plan']]['Plan ID'].tolist()
    plan_ids = current_plan_ids + other_plan_ids
    
    # Create tabs for each plan
    plan_tabs = st.tabs([
        (f"🟢 {plan_id} - {params['plan_names'].get(plan_id, 'Unknown Plan')} (Current)" 
         if summary_df[summary_df['Plan ID'] == plan_id]['Is Current Plan'].item() 
         else f"{plan_id} - {params['plan_names'].get(plan_id, 'Unknown Plan')}") 
        for plan_id in plan_ids
    ])
    
    # Display detailed information for each plan in its tab
    for i, plan_id in enumerate(plan_ids):
        with plan_tabs[i]:
            # Check if this is the current plan
            is_current_plan = summary_df[summary_df['Plan ID'] == plan_id]['Is Current Plan'].item()
            
            display_plan_details(
                plan_id, 
                results['raw_results'].get(plan_id, {}),
                results['coverage_data'].get(plan_id, pd.DataFrame()),
                results['diagnosis_data'].get(plan_id, pd.DataFrame()),
                results['top_member_data'].get(plan_id, pd.DataFrame()),
                params,
                is_current_plan  # Pass the current plan flag
            )

def display_plan_details(plan_id, plan_results, coverage_data, diagnosis_data, top_member_data, params, is_current_plan=False):
    """Display detailed information for a specific plan"""
    if not plan_results:
        st.warning(f"No detailed results available for Plan {plan_id}")
        return
    
    # Extract group results
    group_results = plan_results.get('group_results', {})
    if not group_results:
        st.warning(f"No group results available for Plan {plan_id}")
        return
    
    # Get full plan name
    plan_name = params.get('plan_names', {}).get(plan_id, 'Unknown Plan')
    
    # Plan Summary
    if is_current_plan:
        green_info_block("This is the group's current plan for the selected year.")
    else:
        st.subheader(f"Plan {plan_id} - {plan_name} Summary")
    
    current_plan_data = None
    if not is_current_plan and 'current_plan_id' in params and params['current_plan_id']:
        current_plan_id = params['current_plan_id']
        if ('raw_results' in st.session_state.simulation_results and 
            current_plan_id in st.session_state.simulation_results['raw_results']):
            current_plan_data = st.session_state.simulation_results['raw_results'][current_plan_id]['group_results']

    
    # Create columns for key metrics
    metrics_cols = st.columns(4)
    
    with metrics_cols[0]:
        st.metric(
            "Total Cost", 
            f"${group_results.get('total_member_responsibility', 0) + group_results.get('total_plan_responsibility', 0):,.2f}"
        )
        
    with metrics_cols[1]:
        member_resp = group_results.get('total_member_responsibility', 0)
        
        # If we have current plan data and this isn't the current plan, show delta
        if current_plan_data and not is_current_plan:
            current_member_resp = current_plan_data.get('total_member_responsibility', 0)
            member_delta = member_resp - current_member_resp
            member_delta_pct = (member_delta / current_member_resp * 100) if current_member_resp > 0 else 0
        
            st.metric(
                "Member Responsibility", 
                f"${member_resp:,.2f}",
                f"{member_delta_pct:.1f}%",
                delta_color="inverse"  # Negative is good (cost savings)
            )
        else:
            # Original display without delta
            st.metric(
                "Member Responsibility", 
                f"${member_resp:,.2f}"
            )
    
    with metrics_cols[2]:
        plan_resp = group_results.get('total_plan_responsibility', 0)
        
        # If we have current plan data and this isn't the current plan, show delta
        if current_plan_data and not is_current_plan:
            current_plan_resp = current_plan_data.get('total_plan_responsibility', 0)
            plan_delta = plan_resp - current_plan_resp
            plan_delta_pct = (plan_delta / current_plan_resp * 100) if current_plan_resp > 0 else 0
            
            st.metric(
                "Plan Responsibility", 
                f"${plan_resp:,.2f}",
                f"{plan_delta_pct:.1f}%",
                delta_color="inverse"  # Negative is good (cost savings)
            )
        else:
            # Original display without delta
            st.metric(
                "Plan Responsibility", 
                f"${plan_resp:,.2f}"
            )
        
    with metrics_cols[3]:
        member_count = group_results.get('member_count', 0)
        member_cost = group_results.get('total_member_responsibility', 0)   
        avg_cost = member_cost / member_count if member_count > 0 else 0
        
        # If we have current plan data and this isn't the current plan, show delta
        if current_plan_data and not is_current_plan:
            current_member_count = current_plan_data.get('member_count', 1)
            current_member_cost = current_plan_data.get('total_member_responsibility', 0)
            current_avg_cost = current_member_cost / current_member_count if current_member_count > 0 else 0
            
            avg_delta = avg_cost - current_avg_cost
            avg_delta_pct = (avg_delta / current_avg_cost * 100) if current_avg_cost > 0 else 0
            
            st.metric(
                "Avg Cost/Member", 
                f"${avg_cost:,.2f}",
                f"{avg_delta_pct:.1f}%",
                delta_color="inverse"  # Negative is good (cost savings)
            )
        else:
            # Original display without delta
            st.metric(
                "Avg Cost/Member", 
                f"${avg_cost:,.2f}"
            )
    
    # Create sub-tabs for different analyses
    if params.get('detailed_member_analysis', False):
        cost_tab, coverage_tab, member_tab = st.tabs(["Cost Distribution", "Diagnosis Analysis", "Top Members"])
    else:
        cost_tab, coverage_tab = st.tabs(["Cost Distribution", "Diagnosis Analysis"])
    
    # Cost Distribution Tab
    with cost_tab:
        st.subheader("Cost Distribution Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of member vs plan responsibility
            fig = go.Figure(data=[go.Pie(
                labels=['Member Responsibility', 'Plan Responsibility'],
                values=[
                    group_results.get('total_member_responsibility', 0),
                    group_results.get('total_plan_responsibility', 0)
                ],
                textinfo='value',  # Show actual values
                textposition='outside',  # Move labels outside
                hoverinfo='label+percent+value',
                marker_colors=['#FF6384', '#36A2EB'],
                # Pull out the slices slightly for better visibility
                pull=[0.1, 0]
            )])
            
            fig.update_layout(
                title='Responsibility Distribution',
                height=400,
                margin=dict(l=20, r=20, t=40, b=40),
                # Ensure legend is outside
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"responsibility_pie_{plan_id}")
        with col2:
            st.markdown("<div style='height:125px;'></div>", unsafe_allow_html=True)  # Adjust height as needed
            
            # Add insights about cost distribution
            member_total = group_results.get('total_member_responsibility', 0)
            plan_total = group_results.get('total_plan_responsibility', 0)
            total_cost = member_total + plan_total
            
            green_info_block(f"""**Cost Breakdown**:
            - Total Cost: ${total_cost:,.2f}
            - Member Responsibility: ${member_total:,.2f} ({member_total/total_cost*100:.1f}%)
            - Plan Responsibility: ${plan_total:,.2f} ({plan_total/total_cost*100:.1f}%)
            """, markdown=True)
        
        # Member Distribution by Cost Tier
        def categorize_member_cost(cost):
            """Categorize member cost into fixed price tiers"""
            if cost <= 500:
                return 'Tier 1: $0-$500'
            elif cost <= 2000:
                return 'Tier 2: $500-$2,000'
            else:
                return 'Tier 3: $2,000+'
        
        # High Cost Claims Visualization
        if 'member_results' in plan_results and plan_results['member_results']:
            # Extract member costs
            member_costs = [
                data.get('member_total', 0) + data.get('plan_total', 0) 
                for data in plan_results['member_results'].values()
            ]
            
            import numpy as np
            import pandas as pd
            
            # Categorize members into fixed tiers
            cost_categories = [categorize_member_cost(cost) for cost in member_costs]
            
            # Create DataFrame for analysis
            df_tier = pd.DataFrame({
                'Member Cost': member_costs,
                'Cost Tier': cost_categories
            })
            
            # Analyze cost tiers
            tier_summary = df_tier.groupby('Cost Tier').agg({
                'Member Cost': ['count', 'sum', 'mean']
            })
            tier_summary.columns = ['Members', 'Total Cost', 'Average Cost']
            tier_summary['% of Members'] = tier_summary['Members'] / tier_summary['Members'].sum() * 100
            tier_summary['% of Costs'] = tier_summary['Total Cost'] / tier_summary['Total Cost'].sum() * 100
                
            # Detailed Table and Pie Chart Side by Side
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie Chart of Cost Distribution
                fig_cost_dist = go.Figure(data=[go.Pie(
                    labels=tier_summary.index,
                    values=tier_summary['Total Cost'],
                    textinfo='label+percent',
                    textposition='outside',
                    hoverinfo='label+value+percent',
                    marker_colors=['#2ECC71', '#F39C12', '#E74C3C'],
                    showlegend=False,
                    rotation=270
                )])
                
                fig_cost_dist.update_layout(
                    title='Cost Distribution by Tier',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=40)
                )
                
                st.plotly_chart(fig_cost_dist, use_container_width=True, key=f"cost_tier_pie_chart_{plan_id}")
            
            with col2:
                # Pie Chart of Member Distribution
                fig_member_dist = go.Figure(data=[go.Pie(
                    labels=tier_summary.index,  # Use the cost tier labels
                    values=tier_summary['% of Members'],  # Use the '% of Members' column for values
                    textinfo='label+percent',  # Display labels and percentages
                    textposition='outside',  # Position text outside the pie slices
                    hoverinfo='label+value+percent',  # Show detailed info on hover
                    marker_colors=['#3498DB', '#F39C12', '#E74C3C'],  # Customize slice colors
                    showlegend=False  # Hide legend to avoid redundancy
                )])

                fig_member_dist.update_layout(
                    title='Member Distribution by Tier',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=40)  # Adjust margins to reduce white space
                )

                st.plotly_chart(fig_member_dist, use_container_width=True, key=f"member_dist_pie_chart_{plan_id}")
            
            formatted_summary = tier_summary.style.format({
                'Total Cost': '${:,.2f}',
                'Average Cost': '${:,.2f}',
                '% of Members': '{:.1f}%',
                '% of Costs': '{:.1f}%'
            })
            st.dataframe(formatted_summary, use_container_width=True, key=f"cost_tier_summary__{plan_id}")

    # Diagnosis Analysis Tab
    with coverage_tab:
        st.subheader("Diagnosis Grouper Analysis")
        
        # Add warning about claim line level data
        st.warning("⚠️ This analysis shows diagnosis groupers at the claim line level, not the overall claim level.")
        
        if not diagnosis_data.empty and 'diag_grouper' in diagnosis_data.columns:
            # Create a fresh copy to avoid SettingWithCopyWarning
            diag_data_display = diagnosis_data.copy()
            
            # Format numeric columns before display
            currency_cols = ['member_responsibility', 'plan_responsibility', 'total_cost']
            diag_display_formatted = format_currency(diag_data_display, currency_cols)
            
            diag_formatted = diag_display_formatted.rename(columns={
                'diag_grouper': 'Diagnosis Category',
                'member_responsibility': 'Member Responsibility',
                'plan_responsibility': 'Plan Responsibility',
                'claim_line_number': 'Number of Claim Lines',
                'total_cost': 'Total Cost'
            }).reset_index(drop=True)
            diag_formatted.index = range(1, len(diag_formatted) + 1)
            
            # Display diagnosis data table
            st.dataframe(
                diag_formatted,
                use_container_width=True
            )
            
            # Create bar chart of top diagnosis groupers by cost
            if ('diag_grouper' in diag_data_display.columns and 
                'member_responsibility' in diag_data_display.columns and 
                'plan_responsibility' in diag_data_display.columns):
                
                # Get top 10 diagnosis groupers by total cost
                top_groupers = diag_data_display.head(10)
                
                # Create stacked bar chart
                fig = go.Figure()
                
                # Add member responsibility bars
                fig.add_trace(go.Bar(
                    name='Member Responsibility',
                    x=top_groupers['diag_grouper'],
                    y=top_groupers['member_responsibility'],
                    marker_color='rgba(255, 127, 14, 0.7)',
                    hovertemplate='%{x}<br>$%{y:,.2f}<extra></extra>'
                ))
                
                # Add plan responsibility bars
                fig.add_trace(go.Bar(
                    name='Plan Responsibility',
                    x=top_groupers['diag_grouper'],
                    y=top_groupers['plan_responsibility'],
                    marker_color='rgba(0, 123, 255, 0.7)',
                    hovertemplate='%{x}<br>$%{y:,.2f}<extra></extra>'
                ))
                
                # Update layout
                fig.update_layout(
                    barmode='stack',
                    title='Top 10 Diagnosis Groupers by Cost',
                    xaxis_title='Diagnosis Grouper',
                    yaxis_title='Cost ($)',
                    legend_title='Cost Type',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"bar_diag_groupers_{plan_id}")
                
                # Pie chart of cost distribution by diagnosis grouper
                st.subheader("Percentage of Total Cost by Diagnosis Grouper")
                
                # Make a copy for calculating percentages to avoid SettingWithCopyWarning
                top_groupers_pct = top_groupers.copy()
                top_groupers_pct.loc[:, 'total_pct'] = top_groupers_pct['total_cost'] / top_groupers_pct['total_cost'].sum() * 100
                
                fig = px.pie(
                    top_groupers_pct, 
                    values='total_cost', 
                    names='diag_grouper',
                    title='Cost Distribution by Diagnosis Grouper',
                    hover_data=['total_pct'],
                    labels={'total_cost': 'Total Cost', 'total_pct': 'Percentage'},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                fig.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    hovertemplate='%{label}<br>$%{value:,.2f}<br>%{customdata[0]:.1f}%<extra></extra>'
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True, key=f"pie_diag_distribution_{plan_id}")
        else:
            st.warning("No diagnosis grouper analysis available for this plan.")
    
    # Top Members Tab
    if params.get('detailed_member_analysis', False):
        with member_tab:
            st.subheader("Top Members Analysis")
            
            if not top_member_data.empty:
                # Create a fresh copy to avoid SettingWithCopyWarning
                display_df = top_member_data.copy()
                
                # Calculate percentage of total cost
                total_group_cost = group_results.get('total_member_responsibility', 0) + group_results.get('total_plan_responsibility', 0)
                if total_group_cost > 0:
                    display_df.loc[:, '% of Total Cost'] = (display_df['Total Cost'] / total_group_cost) * 100
                
                # Format numeric columns safely
                formatted_df = format_currency(display_df, ['Member Responsibility', 'Plan Responsibility', 'Total Cost'])
                formatted_df = format_percentage(formatted_df, ['% of Total Cost'])
                
                # Display the table
                st.dataframe(
                    formatted_df,
                    use_container_width=True
                )
                
                # Create a bar chart of top members
                if not display_df.empty:
                    # Bar chart
                    fig = go.Figure()
                    
                    # Add member responsibility bars
                    fig.add_trace(go.Bar(
                        name='Member Responsibility',
                        x=display_df['Member ID'],
                        y=display_df['Member Responsibility'],
                        marker_color='rgba(255, 127, 14, 0.7)',
                        hovertemplate='%{x}<br>$%{y:,.2f}<extra></extra>'
                    ))
                    
                    # Add plan responsibility bars
                    fig.add_trace(go.Bar(
                        name='Plan Responsibility',
                        x=display_df['Member ID'],
                        y=display_df['Plan Responsibility'],
                        marker_color='rgba(0, 123, 255, 0.7)',
                        hovertemplate='%{x}<br>$%{y:,.2f}<extra></extra>'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        barmode='stack',
                        title='Top Members by Cost',
                        xaxis_title='Member',
                        yaxis_title='Cost ($)',
                        legend_title='Cost Type',
                        height=500,
                        xaxis={'categoryorder':'total descending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"bar_top_members_{plan_id}")
                    
                    # Pie chart showing what percentage of total cost comes from top members
                    st.subheader("Percentage of Total Cost from Top Members")
                    
                    top_members_cost = display_df['Total Cost'].sum()
                    other_members_cost = max(0, total_group_cost - top_members_cost)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Pie(
                        labels=['Top Members', 'All Other Members'],
                        values=[top_members_cost, other_members_cost],
                        textinfo='label+percent',
                        insidetextorientation='radial',
                        marker=dict(colors=['rgba(255, 127, 14, 0.7)', 'rgba(0, 123, 255, 0.7)']),
                        hoverinfo='label+value+percent',
                        hovertemplate='%{label}<br>$%{value:,.2f}<br>%{percent:.1f}%<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f"Cost Distribution: Top {len(display_df)} Members vs. All Others",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"pie_top_members_{plan_id}")
                    
                    # Add insight text
                    if len(display_df) > 0:
                        green_info_block(f"The top {len(display_df)} members account for ${top_members_cost:,.2f} " +
                            f"({(top_members_cost/total_group_cost*100) if total_group_cost > 0 else 0:.1f}%) " +
                            f"of the total cost for this plan.")
            else:
                st.warning("No member analysis available for this plan.")

# Run the app
if __name__ == "__main__":
    main()