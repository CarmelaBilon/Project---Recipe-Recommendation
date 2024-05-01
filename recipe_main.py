import streamlit as st
import pandas as pd
import recommender as rcm
import aws_textract_uploader_decoder as awstud
import format_for_display as ffd

st.set_page_config(layout="wide")

# Simulate the top 5 dataframe here
df = pd.DataFrame({
  'rank': [1, 2, 3, 4,5],
  'title': ['Menudo', 'Afritada', 'Sinigang', 'Adobo','Torta'],
  'NER_clean': [["bay","black","chicken","garlic","leaf","pepper","salt","sauce","soy","vinegar","water"],
                ["bay","black","chicken","garlic","leaf","onion","pepper","sauce","soy","vinegar","water"],
                ["bay","chicken","garlic","leaf","onion","sauce","soy","sweet","vinegar"],
                ["bay","black","chicken","leaf","onion","pepper","salt","sauce","soy","vinegar","water"],
                ["bay","black","chicken","garlic","ground","leaf","onion","pepper","sauce","soy","vinegar","water"]],
  'directions': [["Wash the chicken, then put in a pan.", "Add garlic and all the other ingredients.", "Cover and cook for 1/2 hour or until chicken is tender.", "Put over steamed rice to eat."],
                 ["Brown the onion in a frying pan with very little olive or canola oil", "Add chicken and cook at medium heat until brown then add the water, vinegar, soy sauce, black pepper, garlic, bay leaves and hot peppers if you want it spicy", "Cook at high heat until it start boiling then turn down to medium heat until fully cooked."],
                 ["Place onion on bottom of slow cooker with chicken on top of it. In a small bowl, mix the other ingredients and pour over the chicken. Cook on Low for 6 to 8 hours until chicken is very tender."],
                 ["Marinate chicken with salt, pepper and onions about 1/2 hour. Put all ingredients in pan and cook over medium heat until chicken is tender and liquid has cooked down to about half.", "Serve with fluffy rice."],
                 ["In a saucepan, combine the pork, vinegar, garlic, onion, soy sauce, pepper, bay leaves and water and bring to a boil.", "Let cook for about 10 minutes, simmer until the pork is partially cooked.", "Add the chicken, simmer for another 20 minutes until both meat becomes tender and sauce thickens.", "Remove from the heat and serve on a large serving dish with hot steamed rice."]]
})

#recipe_input = "'comp curry powder', 'body wash', 'produce', 'bananas', 'onions red', 'bread artisano white'"

# reco = df

# Setting session state
if 'ings' not in st.session_state:
	st.session_state.ings = 'Ingredients'
     
def set_ings_val(in_str):
    st.session_state.ings = in_str

# This is the uploaded to aws textract
with st.form("Receipt Uploader",clear_on_submit=True):
    st.write("Receipt Uploader to AWS Textract")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

    st.write("Click upload after choosing an image.")
    submitted = st.form_submit_button("Upload")

    if submitted:
        st.write(f'Sending to AWS...')
        # st.write(bytes_data)
        # Set the inputs to the extracted text
        aws_extracted_txt = awstud.main(bytes_data)
        set_ings_val(aws_extracted_txt)
        st.write(f"Extracted items: {aws_extracted_txt}")

with st.form("Recipe Recommender",clear_on_submit=False):
    st.write("Recipe Recommender")
    ings = st.text_input(label='Ingredients', value=st.session_state.ings)
    submitted = st.form_submit_button("Submit")

    if submitted:
        reco = rcm.get_recs(ings)
        # reco = df
        print(reco)
        #st.write("Message is ", msg)
        st.write('\n')
        st.markdown(f'#### *Top Recipes for {ings}* ####')
        # st.table(reco)

        col0, col1, col2, col3, col4 = st.columns(5)

        for x in range(5):
            with eval(f'col{x}'):
                st.header(reco.at[x,'title'])
                st.write('Ingredients:')
                st.write(str(reco.at[x,'NER_clean']))
                st.write('Directions:')
                st.write(ffd.format_for_display(reco.at[x,'directions']))

