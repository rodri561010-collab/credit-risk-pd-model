
#streamlit solo ejecuta app.py 
# 5 components of the expected loss function
# app = titulo -> formulario -> boton -> modelo rf -> resultado

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#cargar el modelo de random forest
modelo=joblib.load("models/random_forest.pkl")

#titulo de la app


st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Credit Risk Prediction Dashboard")

st.caption(
    "Machine Learning • Random Forest • Probability of Default • Expected Loss"
)
st.info(
"""
This application estimates the **Probability of Default (PD)** using a
Random Forest Classifier.

The Expected Loss is calculated using:

EL = PD × LGD × EAD

where:

• PD = Probability of Default

• LGD = Loss Given Default (90%)

• EAD = Exposure at Default
"""
)
# formulario para ingresar los datos
st.write(
""" 
Predice la Probabilidad de Incumplimiento (PD) y 
la Pérdida Esperada utilizando un modelo de Random Forest.
"""
)
# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("Loan Information")

#primer campo de entrada: monto del prestamo
credit_lines = st.sidebar.number_input(
    "Credit lines outstanding",
    min_value=0,
    max_value=20,
    value=2
)
#listo hasta el momento -dataset analizado, modelo rf , modelo guardado , streamlit on , estructura lista

""" 
Haremos una aplicativo tipo banco que prediga el riesgo crediticio de un cliente,
usuario , datos del prestamos , streamlit , dataframe con datos , random forest calculado , expected loss calculado , streamlit muestra el resultado.
"""
# ============================================
# CAMPOS DEL FORMULARIO

#min_value es el valor minimo que puede ingresar el usuario no negativos
#value es el valor por defecto que aparece en el formulario
#step es el incremento que se puede ingresar en el formulario
# ============================================

loan_amt_outstanding = st.sidebar.number_input(
    "Loan Amount Outstanding ($)",
    min_value=0.0,
    value=10000.0,
    step=1000.0
)

total_debt_outstanding= st.sidebar.number_input(
    "Total Debt Outstanding ($)",
    min_value=0.0,
    value=15000.0,
    step=1000.0
)

income = st.sidebar.number_input(
    "Annual Income ($)",
    min_value=0.0,
    value=50000.0,
    step=5000.0
)

years_employed = st.sidebar.number_input(
    "Years Employed",
    min_value=0,
    value=5,
    step=1
)

fico_score = st.sidebar.number_input(
    "FICO Score",
    min_value=300,
    max_value=850,
    value=700
)

#boton prediccion :
predict_button = st.sidebar.button("Generar riesgo crediticio")
if predict_button:
    # Crear un dataframe con los datos ingresados
    nuevo_cliente= pd.DataFrame({
        "credit_lines_outstanding": [credit_lines],
        "loan_amt_outstanding": [loan_amt_outstanding],
        "total_debt_outstanding": [total_debt_outstanding],
        "income": [income],
        "years_employed": [years_employed],
        "fico_score": [fico_score]
    })

    # Realizar la predicción utilizando el modelo de Random Forest
    pd_default=modelo.predict_proba(nuevo_cliente)[0][1]
    #definir el LGD
    recovery_rate = 0.10

    lgd = 1 - recovery_rate
    #calcular la expected loss
    expected_loss = pd_default * lgd * loan_amt_outstanding
    
    st.success("Prediction completed successfully!")

    st.subheader("Prediction Results")
    
    fig = go.Figure(go.Indicator(

    mode="gauge+number",

    value=pd_default*100,

    number={'suffix': "%"},

    title={'text': "Probability of Default"},

    gauge={

        'axis': {'range':[0,100]},

        'bar': {'color': "darkblue"},

        'steps':[

            {'range':[0,10], 'color': "#8BC34A"},     # Verde

            {'range':[10,30], 'color': "#FFC107"},    # Amarillo

            {'range':[30,100], 'color': "#F44336"}    # Rojo

         ]

        }

    ))
    st.plotly_chart(
    fig,
    use_container_width=True
    )
    
    #resultados
    col1, col2,col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Probability of Default",
            f"{pd_default:.2%}"
        )
    
    with col2:
        st.metric(
            "Expected Loss",
            f"${expected_loss:,.2f}"
        )
    with col3:
        st.metric(
            "Loan Amount",
             f"${loan_amt_outstanding:,.2f}"
        )     
    if pd_default <0.10:
            st.success("LOW RISK\n\n El cliente presenta baja probabilidad de perdida, Aprobado")
    elif pd_default < 0.30:
            st.warning("MEDIUM RISK\n\n se recomienda un analisis de credito adicional, Revision")  
    else:
            st.error ("HIGH RISK\n\n Alta probabilidad de perdida,revisar antes de aprobar, Denegado")
            
    st.markdown("---")
    st.subheader("Customer Summary")
    summary = pd.DataFrame({

    "Variable":[

        "Credit Lines",

        "Loan Amount",

        "Total Debt",

        "Annual Income",

        "Years Employed",

        "FICO Score"

    ],

    "Value":[

        credit_lines,

        f"${loan_amt_outstanding:,.2f}",

        f"${total_debt_outstanding:,.2f}",

        f"${income:,.2f}",

        years_employed,

        fico_score

    ]

    })
    st.dataframe(summary, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Feature Importance")
    
    importancias = modelo.feature_importances_   
    feature_importance = pd.DataFrame({

    "Variable":[
        "Credit Lines",
        "Loan Amount",
        "Total Debt",
        "Income",
        "Years Employed",
        "FICO Score"
    ],

    "Importance": importancias

    })
    feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=True
    )
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(
    feature_importance["Variable"],
    feature_importance["Importance"]
    )
    ax.set_title("Feature Importance")

    ax.set_xlabel("Importance")

    ax.set_ylabel("Variables")
    
    st.pyplot(fig)
    
    if st.button("🔄 Reset"):
     st.rerun()
    
    
    
st.markdown("---")

st.markdown("""
### About this project

This application was developed as an end-to-end Machine Learning project.

**Model**
- Random Forest Classifier

**Objective**
- Estimate Probability of Default (PD)
- Calculate Expected Loss (EL)

**Author**
Rodrigo López
""")

#CSS
