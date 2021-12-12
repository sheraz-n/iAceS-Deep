from io import StringIO
from Bio import SeqIO
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import protutil
import base64

icon = Image.open('fav.png')
st.set_page_config(page_title='iAceS-Deep', page_icon = icon)
st.title("iAceS-Deep : A server to identify Acetyle Serine PTM Sites")
alphabet = "XACDEFGHIKLMNOPQRSTUVWY"
myinfo = "Acetylation: A biochemical process that involves the addition of acetyl group to an organic molecule. Examples include the addition of Acetyl Group to Amino Acid Serine during Post Translational Modification (PTM) of Proteins."
myinfo2= "iAceS-Deep server can predict Serine Acetylation sites in protein sequences with increased accuracy using Deep Convolutional Neural Network (CNN) based Classifier."

st.write(myinfo)
st.write(myinfo2)

def seqValidatorProtein (seq):
    alphabet = "XACDEFGHIKLMNOPQRSTUVWY"
    for i in range(len(seq)):
        if (seq[i] not in alphabet):
            return False
    return True    

model = tf.keras.models.load_model('AceSerinCNN21Jul11.h5')
seq = ""
len_seq = 0
image = Image.open('methComplete.png')
#st.subheader("""iCarboxE-Deep""")
caption= "The proposed methodology to develop iAceS-Deep Classifier"
st.image(image, use_column_width=True, caption=caption)

st.sidebar.subheader(("Input Sequence(s) (Text FORMAT ONLY)"))
#fasta_string  = st.sidebar.text_area("Sequence Input", height=200)
seq_string = st.sidebar.text_area("Sequence Input", height=200)          
st.subheader("Click the Example Button for Sample Data")

if st.button('Example'):
    st.info("Protein Sequences with AcetylSerine Sites")
    st.code("MSMILSASVIRVRDGLPLSASTDYEQSTGMQECRKYFKMLSRKLAQLPDRCTLKTGHYNINFISSLGVSYMMLCTENYPNVLAFSFLDELQKEFITTYNMMKTNTAVRPYCFIEFDNFIQRTKQRYNNPRSLSTKINLSDMQTEIKLRPPYQISMCELGSANGVTSAFSVDCKGAGKISSAHQRLEPATLSGIVGFILSLLCGALNLIRGFHAIESLLQSDGDDFNYIIAFFLGTAACLYQCYLLVYYTGWRNVKSFLTFGLICLCNMYLYELRNLWQLFFHVTVGAFVTLQIWLRQAQGKAPDYDV", language="markdown")
    st.code("MSSEAETQQPPAAPPAAPALSAADTKPGTTGSGAGSGGPGGLTSAAPAGGDKKVIATKVLGTVKWFNVRNGYGFINRNDTKEDVFVHQTAIKKNNPRKYLRSVGDGETVEFDVVEGEKGAEAANVTGPGGVPVQGSKYAADRNHYRRYPRRRGPPRNYQQNYQNSESGEKNEGSESAPEGQAQQRRPYRRRRFPPYYMRRPYGRRPQYSNPPVQGEVMEGADNQGAGEQGRPVRQNMYRGYRPRFRRGPPRQRQPREDGNEEDKENQGDETQGQQPPQRRYRRNFNYRRRRPENPKPQDGKETKAADPPAENSSAPEAEQGGAE", language="markdown")
    st.code("MSELEKAVVALIDVFHQYSGREGDKHKLKKSELKELINNELSHFLEEIKEQEVVDKVMETLDSDGDGECDFQEFMAFVAMITTACHEFFEHE", language="markdown")
    st.code("MSSNNSGLSAAGEIDESLYSRQLYVLGKEAMLKMQTSNVLILGLKGLGVEIAKNVVLAGVKSMTVFDPEPVQLADLSTQFFLTEKDIGQKRGDVTRAKLAELNAYVPVNVLDSLDDVTQLSQFQVVVATDTVSLEDKVKINEFCHSSGIRFISSETRGLFGNTFVDLGDEFTVLDPTGEEPRTGMVSDIEPDGTVTMLDDNRHGLEDGNFVRFSEVEGLDKLNDGTLFKVEVLGPFAFRIGSVKEYGEYKKGGIFTEVKVPRKISFKSLKQQLSNPEFVFSDFAKFDRAAQLHLGFQALHQFAVRHNGELPRTMNDEDANELIKLVTDLSVQQPEVLGEGVDVNEDLIKELSYQARGDIPGVVAFFGGLVAQEVLKACSGKFTPLKQFMYFDSLESLPDPKNFPRNEKTTQPVNSRYDNQIAVFGLDFQKKIANSKVFLVGSGAIGCEMLKNWALLGLGSGSDGYIVVTDNDSIEKSNLNRQFLFRPKDVGKNKSEVAAEAVCAMNPDLKGKINAKIDKVGPETEEIFNDSFWESLDFVTNALDNVDARTYVDRRCVFYRKPLLESGTLGTKGNTQVIIPRLTESYSSSRDPPEKSIPLCTLRSFPNKIDHTIAWAKSLFQGYFTDSAENVNMYLTQPNFVEQTLKQSGDVKGVLESISDSLSSKPHNFEDCIKWARLEFEKKFNHDIKQLLFNFPKDAKTSNGEPFWSGAKRAPTPLEFDIYNNDHFHFVVAGASLRAYNYGIKSDDSNSKPNVDEYKSVIDHMIIPEFTPNANLKIQVNDDDPDPNANAANGSDEIDQLVSSLPDPSTLAGFKLEPVDFEKDDDTNHHIEFITACSNCRAQNYFIETADRQKTKFIAGRIIPAIATTTSLVTGLVNLELYKLIDNKTDIEQYKNGFVNLALPFFGFSEPIASPKGEYNNKKYDKIWDRFDIKGDIKLSDLIEHFEKDEGLEITMLSYGVSLLYASFFPPKKLKERLNLPITQLVKLVTKKDIPAHVSTMILEICADDKEGEDVEVPFITIHL", language="markdown")
    
    
    #st.info("Non-ORI Sequences")

if st.sidebar.button("PREDICT"):
    if (seq_string==""):
        st.error("Please input the sequence first")
        exit()
    
    #validation of proper protein string
    if (not seqValidatorProtein(seq_string)):
        st.error("Invalid Protein Sequence detected. Remove numerics and special characters")
        exit()
    
    #Symbol of Amino Acid needs to be changed Here     
    amino_acid = "S"
    sample_list, idx_list = protutil.samplesfromProtein(amino_acid, seq_string)
    tmp =[protutil.encode_sample(a) for a in sample_list]
    X = np.array(tmp)
    y_pred_cont = model.predict(X)
    y_pred = np.where (y_pred_cont < 0.52, 0, 1 )
    tmp = np.nonzero(y_pred)[0]
    pos_site_list = [idx_list[a] for a in tmp]
    pos_seq_list = [sample_list[a] for a in tmp]
    #pos_prob_list = [y_pred_cont[a] for a in tmp]
    pos_prob_list = ["{:.3f}".format(float(y_pred_cont[a])) for a in tmp]
    
    assert(len(pos_site_list) == len(pos_prob_list))
    assert(len(pos_site_list) == len(pos_seq_list))
    
    final_df = pd.DataFrame({#'Sequence ID': range(len(pos_site_list)),
                             'Positive Site Sequence': pos_seq_list,
                             'Site_index': pos_site_list,
                             '+ve Probability': pos_prob_list
                             })
              
    st.table(final_df) #does not truncate columns

datafile = "Dataset.7z"    
if st.sidebar.button("Download Dataset"):
    with open (datafile, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download=\'{datafile}\'>\
        Click to download\
        </a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)    
        
   
