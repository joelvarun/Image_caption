import streamlit as st
from model import predict_step

# Headline
st.write("# Image Captator")

# Project explanation
st.header("Joel")
st.write("")

# Data preparation



#show the picture and generate the caption
def gen_caption(picture):
    st.image(picture)
    st.subheader('Generated caption:')
    with st.spinner(text='This may take a moment...'):
        caption = predict_step([picture])
    st.write(caption[0])

    
#user chooses between preuploaded picture or uploads one himself
col1, col2, col3 = st.columns([0.5,1,0.5])
with col1:
    pass
with col2:
    st.subheader('Caption Generator')
    user_choice = st.radio(label='Choose from either option', options=['Upload your own picture','Image from our dataset'])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
with col3:
    pass


if user_choice == 'Upload your own picture':
    picture = st.file_uploader('', type=['jpg'])
    if picture != None:
        gen_caption(picture)
else:
    #center the button
    with col1:
        pass
    with col2:
        #get a random picture from out dataset
        if st.button("Get a picture from our dataset:"):
            picture = '1200px-Almeida_Júnior_-_Saudade_(Longing)_-_Google_Art_Project.jpg'
            #x = random.randint(0,500)
            #picture = list(features.keys())[x]
            #picture = features[pic].reshape((1,2048))
            #picture = plt.imread(images + pic)
            gen_caption(picture)
    with col3:
        pass
    
