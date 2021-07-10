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
        submitted = st.form_submit_button('Submit and Clean rewards')
        # if submitted:
        arr = init_matrix(x_size,y_size, startpoint, destination)
    show_heatmap(arr)
    col1, col2 = st.beta_columns(2)
    iterations = col2.slider('Iterations', 1, 50, 1)
    if col1.button('Find Destination'):
        for i in  range(iterations):
            st.header('Iteration '+str(i+1))
            arr = find_dest(startpoint, destination, x_size, y_size, arr, i)
        st.header('Final Run')
        arr = find_dest(startpoint, destination, x_size, y_size, arr, iterations, rand=False)

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

def find_dest(start, dest, x,y, df, run_number, rand=True):
    # df = pd.DataFrame(0, index=range(0,y),columns=range(0,x)).astype(float)
    x_start = start[0]
    y_start = start[1]
    dest_x = dest[0]
    dest_y = dest[1]
    # # st.write(df)
    df.at[x_start,y_start] = 0
    if run_number==0:
        df.at[dest_x,dest_y] = 0
    # st.write(df)
    step_list = []
    i = 0
    actual_point = start
    actual_point_x, actual_point_y = start[0], start[1]
    #st.write(df.transpose())
    while (actual_point!=dest)&(i<5000):
        step_possabilities = ['u', 'd', 'l', 'r']
        if rand==False:
            neighbors = find_neighbors(actual_point, df, y, x)# get next points to actual point
            if all(v[0] == 0 for v in neighbors):
                print(True)
                next_step = step_possabilities[random.randint(0,3)]
            else:
                neighbors.sort(key=lambda tup: tup[0], reverse=True)
                next_step = neighbors[0][1]
        else:
            next_step = step_possabilities[random.randint(0,3)]
        print(next_step)
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
    # st.dataframe(step_df)
    plc_dev_heatmap = st.empty()
    rewarded_points = []
    for i in reversed(range(len(step_df))):
        # if step_df.at[i, 'reward']>df.at[step_df.at[i,'x'], step_df.at[i,'y']]:
        if (step_df.at[i,'x'], step_df.at[i,'y']) not in rewarded_points:
            df.at[step_df.at[i,'x'], step_df.at[i,'y']] += step_df.at[i, 'reward']
            rewarded_points.append((step_df.at[i,'x'], step_df.at[i,'y']))
        # if i%100==0:
        #     print(i)
    # plc_dev_heatmap = st.empty()
    if rand == False:
        show_heatmap_v2(df, plc_dev_heatmap, step_df, start, dest)
    # st.write(df)
    return df

def show_heatmap(matrix):
    fig = plt.figure(figsize=(10,8))
    r = sns.heatmap(matrix.transpose(), linewidths=3, cmap='Blues', cbar=False)
    st.pyplot(fig)

def show_heatmap_v2(matrix, plc,step_df, start, dest):
    fig = plt.figure(figsize=(10,8))
    r = sns.heatmap(matrix.transpose(), linewidths=3, cmap='Blues', cbar=False, mask=(matrix.transpose()==0), annot=True)
    r.add_patch(plt.Rectangle(start,1,1,fill=False, edgecolor='red', lw=5))
    for i in range(len(step_df)):
        x = step_df.at[i,'x']
        y = step_df.at[i, 'y']
        r.add_patch(plt.Rectangle((x,y),1,1,fill=False, edgecolor='black', lw=5))
    r.add_patch(plt.Rectangle(dest,1,1,fill=False, edgecolor='green', lw=5))
    #ax.add_patch(plt.Rectangle((1,2),2,1,fill=False, edgecolor='black', lw=5))
    plc.pyplot(fig)

def find_neighbors(actual_point, df, y,x):
    actual_point_x, actual_point_y = actual_point[0], actual_point[1]
    neighbors= []
    if actual_point_y!=y-1:
        # try:
        lower_point = df.at[actual_point_x, actual_point_y+1]
        neighbors.append((lower_point, 'd'))
        # except:
        #     pass
    if actual_point_y!=0:
        # try:
        upper_point = df.at[actual_point_x, actual_point_y-1]
        neighbors.append((upper_point, 'u'))
        # except:
        #     pass
    if actual_point_x!=x-1:
        # try:
        right_point = df.at[actual_point_x+1, actual_point_y]
        neighbors.append((right_point, 'r'))
        # except:
        #     pass
    if actual_point_x !=0:
        # try:
        left_point = df.at[actual_point_x-1, actual_point_y]
        neighbors.append((left_point, 'l'))
    return neighbors

main()