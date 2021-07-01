import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time

@st.cache(allow_output_mutation=True)
def default():
    rng = np.random.default_rng()
    df = pd.DataFrame(rng.integers(0, 1, size=(10, 10))).astype(float)
    x_start = 9
    y_start = 9
    dest_x = 0
    dest_y = 0
    df.at[x_start,y_start] = 1
    df.at[dest_x,dest_y] = 1
    return df

def main():
    arr = default()
    st.header('Reinforcement Learning')
    with st.sidebar.form('init_vars'):
        st.header('Initialize Field')
        x_size = st.slider('x_size', 1, 20, 10)
        y_size = st.slider('y_size', 1, 20, 10)
        dest_x = st.number_input('Destination-X', 0, x_size-1 ,0)
        dest_y = st.number_input('Destination-Y', 0, y_size-1,0)
        x_start = st.number_input('Start-X', 0, x_size-1 ,x_size-1)
        y_start = st.number_input('Start-Y', 0, y_size-1, y_size-1)
        startpoint = (x_start,y_start)
        destination = (dest_x,dest_y)
        # if st.checkbox('Random Startpoint', True):
        #     x_start = random.randint(0,x_size-1)
        #     y_start = random.randint(0,y_size-1)
        #     startpoint = (x_start,y_start)
        #     while startpoint==destination:
        #         x_start = random.randint(0,x_size-1)
        #         y_start = random.randint(0,y_size-1)
        #         startpoint = (x_start,y_start)
        submitted = st.form_submit_button('Submit or Clean rewards')
        # if submitted:
        arr = init_matrix(x_size,y_size, startpoint, destination)
    show_heatmap(arr)
    col1, col2 = st.beta_columns(2)
    iterations = col2.slider('Iterations', 1, 15, 1)
    if col1.button('Find Destination'):
        for i in  range(iterations):
            arr = find_dest(startpoint, destination, x_size, y_size, arr)

    # show_heatmap(arr)

def init_matrix(x,y, startpoint, destination):
    rng = np.random.default_rng()
    df = pd.DataFrame(rng.integers(0, 1, size=(x, y))).astype(float)
    x_start = startpoint[0]
    y_start = startpoint[1]
    dest_x = destination[0]
    dest_y = destination[1]
    df.at[x_start,y_start] = 1
    df.at[dest_x,dest_y] = 1
    return df

def find_dest(start, dest, x,y, df):
    # df = pd.DataFrame(0, index=range(0,y),columns=range(0,x)).astype(float)
    # x_start = start[0]
    # y_start = start[1]
    # dest_x = dest[0]
    # dest_y = dest[1]
    # # st.write(df)
    # df.at[x_start,y_start] = 1
    # df.at[dest_x,dest_y] = 1
    # st.write(df)
    step_list = []
    i = 0
    actual_point = start
    actual_point_x, actual_point_y = start[0], start[1]
    while (actual_point!=dest)&(i<2000):
        step_possabilities = ['u', 'd', 'l', 'r']
        next_step = step_possabilities[random.randint(0,3)]
        if next_step == 'u':
            if actual_point_y !=0:
                actual_point_y -=1
        elif next_step == 'd':
            if actual_point_y !=y-1:
                actual_point_y +=1
        elif next_step =='l':
            if actual_point_x !=0:
                actual_point_x -=1
        elif next_step =='r':
            if actual_point_x !=x-1:
                actual_point_x  +=1
        if actual_point == (actual_point_x,actual_point_y):
            continue
        i+=1
        step = (i, actual_point_x, actual_point_y)
        print(step)
        step_list.append(step)
        actual_point = (actual_point_x, actual_point_y)
    step_df = pd.DataFrame(step_list,columns=['step', 'x', 'y'])
    st.write('It needed '+str(i)+' Steps.')
    step_df['reward'] = 0.9**abs(step_df['step']-len(step_df))
    st.dataframe(step_df)
    plc_dev_heatmap = st.empty()
    for i in range(len(step_df)):
        if step_df.at[i, 'reward']>df.at[step_df.at[i,'x'], step_df.at[i,'y']]:
            df.at[step_df.at[i,'x'], step_df.at[i,'y']] = step_df.at[i, 'reward']
        # if i%100==0:
        #     print(i)
    # plc_dev_heatmap = st.empty()
    show_heatmap_v2(df, plc_dev_heatmap)
    st.write(df)
    return df

def show_heatmap(matrix):
    fig = plt.figure(figsize=(10,8))
    r = sns.heatmap(matrix.transpose(), linewidths=3, cmap='Blues', cbar=False)
    st.pyplot(fig)

def show_heatmap_v2(matrix, plc):
    fig = plt.figure(figsize=(10,8))
    r = sns.heatmap(matrix.transpose(), linewidths=3, cmap='Blues', cbar=False, mask=(matrix.transpose()==0))
    plc.pyplot(fig)

main()