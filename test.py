import numpy as np
import pandas as pd
import streamlit as st

# A text
st.text('Random Numbers will change every time the script reruns')

st.text('Let’s create a data frame and change its formatting with') 
st.text('a Pandas Styler object. In this example, you’ll use Numpy to generate a random sample,')
st.text('and the st.dataframe() method to draw an interactive table.')

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))

st.text('Streamlit also has a method for static table generation: st.table().')

dataframe = pd.DataFrame(
    np.random.randn(1, 10),
    columns=('col %d' % i for i in range(10)))
st.table(dataframe)


st.header('Widgets')
st.text('When you’ve got the data or model into the state that you want to explore,')
st.text('you can add in widgets like st.slider(), st.button() or st.selectbox().')
st.text('It’s really straightforward — treat widgets as variables')
st.code('x = st.slider(x)')
st.code('st.write(x, squared is, x * x)')

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

st.text('On first run, the app above should output the text “0 squared is 0”.')
st.text('Then every time a user interacts with a widget, Streamlit simply reruns your')
st.text('script from top to bottom, assigning the current state of the widget to your variable in the process.')
st.text('For example, if the user moves the slider to position 10, Streamlit will rerun the code above and set ')
st.text('x to 10 accordingly. So now you should see the text “10 squared is 100”.')

st.header('Layout')
st.text('Streamlit makes it easy to organize your widgets in a left panel sidebar with st.sidebar.')
st.text('Each element that’s passed to st.sidebar is pinned to the left, allowing users to focus')
st.text('on the content in your app while still having access to UI controls.')

st.text('For example, if you want to add a selectbox and a slider to a sidebar, use st.sidebar.slider')
st.text('and st.siderbar.selectbox instead of st.slider and st.selectbox:')

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

st.text('Beyond the sidebar, Streamlit offers several other ways to control the layout of your app.')
st.text('st.beta_columns lets you place widgets side-by-side,')
st.text('and st.beta_expander lets you conserve space by hiding away large content.')

left_column, right_column = st.beta_columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")

st.header('Caching')
st.text('The Streamlit cache allows your app to execute quickly even when loading data from the web,') 
st.text('manipulating large datasets, or performing expensive computations.')

st.text('To use the cache, wrap functions with the @st.cache decorator:')

st.code('@st.cache')  # 👈 This function will be cached
st.code('def my_slow_function(arg1, arg2):')
st.code('Do something really slow in here!') 
st.code('return the_output')

#When you mark a function with the @st.cache decorator,
# it tells Streamlit that whenever the function is called it needs to check a few things:
#The input parameters that you called the function with
#The value of any external variable used in the function
#The body of the function
#The body of any function used inside the cached function