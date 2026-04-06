import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Trading Monitor", layout="wide")

# Title
st.title("📈 Trading Monitor - IMC Prosperity")

# Load data functions
@st.cache_data
def load_prices_data(day):
    """Load price/order book data for a specific day"""
    file_path = f'data/prices_round_0_day_{day}.csv'
    df = pd.read_csv(file_path, sep=';')
    return df

@st.cache_data
def load_trades_data(day):
    """Load trades data for a specific day"""
    file_path = f'data/trades_round_0_day_{day}.csv'
    
    # Debug: Check if file exists
    import os
    if not os.path.exists(file_path):
        st.error(f"❌ File not found: {file_path}")
        st.info(f"📁 Looking for file at: {os.path.abspath(file_path)}")
        return None
    
    try:
        # Try with semicolon separator first
        df = pd.read_csv(file_path, sep=';')
        
        if df.empty:
            st.warning(f"⚠️ File {file_path} is empty")
            return None
            
        # Check if we have the expected columns
        expected_cols = ['timestamp', 'symbol', 'price', 'quantity']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"⚠️ Missing columns: {missing_cols}")
            st.info(f"📊 Found columns: {list(df.columns)}")
            
            # Try comma separator instead
            df = pd.read_csv(file_path, sep=',')
            st.success("✅ Loaded with comma separator instead")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading {file_path}: {str(e)}")
        st.info(f"💡 Try checking the file format and separator")
        return None

# Day selector
st.sidebar.markdown("## 📅 Settings")
selected_day = st.sidebar.selectbox(
    "Select Day:",
    options=[-1, -2],
    format_func=lambda x: f"Day {x}" if x < 0 else f"Day +{x}"
)

# Load data for selected day
df_prices = load_prices_data(selected_day)
df_trades = load_trades_data(selected_day)

# Function to convert timestamp to time format HH:MM:SS
def timestamp_to_time(ts, max_ts):
    """Convert timestamp to HH:MM:SS format"""
    # Map timestamp range to 24 hours (0 to 86399 seconds = 00:00:00 to 23:59:59)
    seconds_in_day = 86400
    seconds = int((ts / max_ts) * (seconds_in_day - 1))  # -1 to max at 23:59:59
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Function to render order book for a product
def render_order_book(product_name, row_data):
    st.markdown(f"## {product_name}")
    
    # Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Mid Price", f"{row_data['mid_price']:.2f}")
    with col_m2:
        bid_1 = row_data['bid_price_1']
        ask_1 = row_data['ask_price_1']
        if pd.notna(bid_1) and pd.notna(ask_1):
            spread = float(ask_1) - float(bid_1)
            st.metric("Spread", f"{spread:.0f}")
        else:
            st.metric("Spread", "N/A")
    with col_m3:
        st.metric("P&L", f"{row_data['profit_and_loss']:.2f}")
    
    st.markdown("---")
    
    # Order Book Table
    col_bid, col_ask = st.columns(2)
    
    with col_bid:
        st.markdown("### 🔴 BID")
        bid_data = []
        for i in range(1, 4):  # Always show 3 levels
            price_col = f'bid_price_{i}'
            volume_col = f'bid_volume_{i}'
            
            if price_col in row_data.index and volume_col in row_data.index:
                price = row_data[price_col]
                volume = row_data[volume_col]
                
                if pd.notna(price) and pd.notna(volume) and price != '' and volume != '':
                    bid_data.append({
                        'Level': i,
                        'Price': f"{int(price):,}",
                        'Volume': int(volume)
                    })
                else:
                    bid_data.append({
                        'Level': i,
                        'Price': '-',
                        'Volume': '-'
                    })
            else:
                bid_data.append({
                    'Level': i,
                    'Price': '-',
                    'Volume': '-'
                })
        
        bid_df = pd.DataFrame(bid_data)
        st.dataframe(bid_df, hide_index=True, use_container_width=True)
    
    with col_ask:
        st.markdown("### 🟢 ASK")
        ask_data = []
        for i in range(1, 4):  # Always show 3 levels
            price_col = f'ask_price_{i}'
            volume_col = f'ask_volume_{i}'
            
            if price_col in row_data.index and volume_col in row_data.index:
                price = row_data[price_col]
                volume = row_data[volume_col]
                
                if pd.notna(price) and pd.notna(volume) and price != '' and volume != '':
                    ask_data.append({
                        'Level': i,
                        'Price': f"{int(price):,}",
                        'Volume': int(volume)
                    })
                else:
                    ask_data.append({
                        'Level': i,
                        'Price': '-',
                        'Volume': '-'
                    })
            else:
                ask_data.append({
                    'Level': i,
                    'Price': '-',
                    'Volume': '-'
                })
        
        ask_df = pd.DataFrame(ask_data)
        st.dataframe(ask_df, hide_index=True, use_container_width=True)
    
    # Chart
    st.markdown("#### 📊 Order Book Visualization")
    
    bid_prices = []
    bid_volumes = []
    ask_prices = []
    ask_volumes = []
    
    for i in range(1, 4):
        bid_price = row_data[f'bid_price_{i}']
        bid_volume = row_data[f'bid_volume_{i}']
        ask_price = row_data[f'ask_price_{i}']
        ask_volume = row_data[f'ask_volume_{i}']
        
        if pd.notna(bid_price) and pd.notna(bid_volume) and bid_price != '' and bid_volume != '':
            bid_prices.append(float(bid_price))
            bid_volumes.append(float(bid_volume))
        
        if pd.notna(ask_price) and pd.notna(ask_volume) and ask_price != '' and ask_volume != '':
            ask_prices.append(float(ask_price))
            ask_volumes.append(float(ask_volume))
    
    if bid_prices or ask_prices:
        fig = go.Figure()
        
        # Bids (red bars, to the left)
        if bid_prices:
            fig.add_trace(go.Bar(
                y=bid_prices,
                x=[-v for v in bid_volumes],
                orientation='h',
                name='Bid',
                marker_color='rgb(239, 68, 68)',
                text=bid_volumes,
                textposition='auto',
            ))
        
        # Asks (green bars, to the right)
        if ask_prices:
            fig.add_trace(go.Bar(
                y=ask_prices,
                x=ask_volumes,
                orientation='h',
                name='Ask',
                marker_color='rgb(34, 197, 94)',
                text=ask_volumes,
                textposition='auto',
            ))
        
        fig.update_layout(
            xaxis_title='Volume',
            yaxis_title='Price',
            barmode='overlay',
            height=350,
            showlegend=True,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data to display order book chart")

# Timestamp selector
min_ts = int(df_prices['timestamp'].min())
max_ts = int(df_prices['timestamp'].max())

timestamp = st.slider(
    "Select Timestamp:",
    min_value=min_ts,
    max_value=max_ts,
    value=min_ts,
    step=100
)

# Display current time
st.markdown(f"### ⏰ {timestamp_to_time(timestamp, max_ts)}")
st.markdown("---")

# Create tabs for different views
tab1, tab2 = st.tabs(["📊 Order Book", "💱 Executed Trades"])

with tab1:
    # ORDER BOOK VIEW
    # Get data for both products at selected timestamp
    products = ['TOMATOES', 'EMERALDS']
    product_data = {}

    for product in products:
        df_product = df_prices[df_prices['product'] == product]
        current_row = df_product[df_product['timestamp'] == timestamp]
        
        if len(current_row) == 0:
            # If exact timestamp doesn't exist, find nearest
            idx = (df_product['timestamp'] - timestamp).abs().idxmin()
            current_row = df_product.loc[idx]
        else:
            current_row = current_row.iloc[0]
        
        product_data[product] = current_row
    
    # Display both products side-by-side
    col1, col2 = st.columns(2)

    with col1:
        render_order_book('TOMATOES', product_data['TOMATOES'])

    with col2:
        render_order_book('EMERALDS', product_data['EMERALDS'])

with tab2:
    # TRADES VIEW
    st.markdown("### Executed Trades")
    
    if df_trades is not None:
        # Filter trades up to current timestamp
        trades_filtered = df_trades[df_trades['timestamp'] <= timestamp].copy()
        
        if len(trades_filtered) > 0:
            # Overall metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric("Total Trades", len(trades_filtered))
            
            with col_m2:
                total_volume = trades_filtered['quantity'].sum()
                st.metric("Total Volume", f"{int(total_volume):,}")
            
            with col_m3:
                avg_price = trades_filtered['price'].mean()
                st.metric("Avg Price", f"{avg_price:,.2f}")
            
            with col_m4:
                total_value = (trades_filtered['price'] * trades_filtered['quantity']).sum()
                st.metric("Total Value", f"{total_value:,.0f}")
            
            st.markdown("---")
            
            # Product breakdown
            col_prod1, col_prod2 = st.columns(2)
            
            for idx, product in enumerate(['TOMATOES', 'EMERALDS']):
                product_trades = trades_filtered[trades_filtered['symbol'] == product].copy()
                
                with col_prod1 if idx == 0 else col_prod2:
                    st.markdown(f"#### {product}")
                    
                    if len(product_trades) > 0:
                        # Product metrics
                        col_p1, col_p2, col_p3 = st.columns(3)
                        
                        with col_p1:
                            st.metric("Trades", len(product_trades))
                        
                        with col_p2:
                            st.metric("Volume", f"{int(product_trades['quantity'].sum()):,}")
                        
                        with col_p3:
                            st.metric("Avg Price", f"{product_trades['price'].mean():,.2f}")
                        
                        # Price chart over time
                        fig = go.Figure()
                        
                        # Convert timestamps to time for x-axis
                        product_trades['time'] = product_trades['timestamp'].apply(
                            lambda x: timestamp_to_time(x, max_ts)
                        )
                        
                        # Scatter plot of trades
                        fig.add_trace(go.Scatter(
                            x=product_trades['time'],
                            y=product_trades['price'],
                            mode='markers',
                            marker=dict(
                                size=product_trades['quantity'] * 2,
                                color=product_trades['price'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Price")
                            ),
                            text=product_trades.apply(
                                lambda row: f"Time: {row['time']}<br>Price: {row['price']}<br>Qty: {row['quantity']}", 
                                axis=1
                            ),
                            hovertemplate='%{text}<extra></extra>',
                            name='Trades'
                        ))
                        
                        # Add moving average
                        if len(product_trades) > 5:
                            product_trades_sorted = product_trades.sort_values('timestamp')
                            product_trades_sorted['ma_price'] = product_trades_sorted['price'].rolling(window=5).mean()
                            
                            fig.add_trace(go.Scatter(
                                x=product_trades_sorted['time'],
                                y=product_trades_sorted['ma_price'],
                                mode='lines',
                                line=dict(color='red', width=2),
                                name='MA(5)'
                            ))
                        
                        fig.update_layout(
                            xaxis_title='Time',
                            yaxis_title='Price',
                            height=300,
                            showlegend=True,
                            margin=dict(l=20, r=20, t=20, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Recent trades table
                        st.markdown("##### Recent Trades")
                        recent_trades = product_trades.sort_values('timestamp', ascending=False).head(10)
                        recent_trades['time'] = recent_trades['timestamp'].apply(
                            lambda x: timestamp_to_time(x, max_ts)
                        )
                        
                        display_trades = recent_trades[['time', 'price', 'quantity', 'symbol']].copy()
                        st.dataframe(
                            display_trades,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info(f"No {product} trades yet")
            
        else:
            st.info("No trades executed yet at this timestamp")
    else:
        st.warning(f"No trades data available for Day {selected_day}")
