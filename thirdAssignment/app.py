"""Streamlit frontend for the churn API.
  run with streamlit run app.py
  Should run after the API. 
"""

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

API = 'http://127.0.0.1:5000'

st.set_page_config(page_title='Churn predictor', layout='wide')


def get(path):
    return requests.get(f'{API}{path}', timeout=10).json()


@st.cache_data
def get_schema():
    """Static for the life of the server, so it is fetched once."""
    return get('/schema')


@st.cache_data
def get_importance():
    return get('/importance')


try:
    schema_response = get_schema()
except requests.exceptions.RequestException:
    st.error(f'No answer from the API at {API}. Start it with `python api.py`.')
    st.stop()

schema = schema_response['schema']

#threshold

st.sidebar.title('Decision threshold')
st.sidebar.caption(
    'The probability above which a customer is called a churn risk. '
    'It lives on the server, so changing it here changes it for everyone.')

threshold_state = get('/threshold')
suggested = threshold_state['suggested']

if 'pending' in st.session_state:
    current = st.session_state.pop('pending')
    requests.post(f'{API}/threshold', json={'threshold': current}, timeout=10)
else:
    current = threshold_state['threshold']

chosen = st.sidebar.slider('Threshold', 0.0, 1.0, float(current), 0.005)
if abs(chosen - current) > 1e-9:
    requests.post(f'{API}/threshold', json={'threshold': chosen}, timeout=10)
    current = chosen


def jump_to(value):
    st.session_state['pending'] = float(value)


st.sidebar.write('Suggested values:')
st.sidebar.button(
    f"Best F1  ({suggested['best_f1']['threshold']:.3f})",
    width='stretch',
    on_click=jump_to, args=(suggested['best_f1']['threshold'],))
st.sidebar.button(
    f"Lowest cost  ({suggested['best_cost']['threshold']:.3f})",
    width='stretch',
    on_click=jump_to, args=(suggested['best_cost']['threshold'],))
st.sidebar.button(
    'Default  (0.500)',
    width='stretch',
    on_click=jump_to, args=(0.5,))
st.sidebar.caption(
    f"Lowest cost assumes losing a customer costs "
    f"{suggested['best_cost']['false_negative_cost']}x a wasted discount.")

#layout

st.title('Customer churn predictor')
st.caption(f"{schema_response['model']}  ·  threshold {current:.3f}")

predict_tab, results_tab, importance_tab = st.tabs(
    ['Predict', 'Test results', 'Feature importance'])

#predict

with predict_tab:
    st.subheader('One customer')
    st.caption('Values are sent as they are. The API holds the encoder, so '
               'no manual normalisation is needed.')

    inputs = {}
    columns = st.columns(3)
    for i, (name, spec) in enumerate(schema.items()):
        with columns[i % 3]:
            if spec['kind'] == 'number':
                inputs[name] = st.number_input(
                    name, min_value=spec['min'], max_value=spec['max'],
                    value=spec['default'])
            else:
                values = spec['values']
                inputs[name] = st.selectbox(
                    name, values, index=values.index(spec['default']))

    if st.button('Predict', type='primary'):
        answer = requests.get(f'{API}/predict', params=inputs, timeout=10).json()
        if 'error' in answer:
            st.error(answer['error'])
        else:
            left, right = st.columns(2)
            left.metric('Churn probability', f"{answer['probability']:.1%}")
            right.metric('Decision',
                         'Churn risk' if answer['churn'] else 'Not flagged')
            st.progress(answer['probability'])
            st.caption(f"Flagged when the probability reaches "
                       f"{answer['threshold']:.3f}.")

#test results

with results_tab:
    metrics = get('/metrics')
    st.subheader(f"Test set at threshold {metrics['threshold']:.3f}")

    cells = st.columns(4)
    for cell, key in zip(cells, ['accuracy', 'precision', 'recall', 'f1']):
        cell.metric(key.capitalize(), f'{metrics[key]:.3f}')

    left, right = st.columns(2)

    with left:
        confusion = metrics['confusion']
        figure, axes = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(
            pd.DataFrame([[confusion['tn'], confusion['fp']],
                          [confusion['fn'], confusion['tp']]]).values,
            display_labels=['No churn', 'Churn']).plot(ax=axes, colorbar=False)
        axes.set_title('Confusion matrix')
        st.pyplot(figure)
        st.caption(
            f"{confusion['fn']} churning customers missed, "
            f"{confusion['fp']} offers wasted on customers who were staying. "
            f"{metrics['flagged']} customers flagged in total.")

    with right:
        curve = pd.DataFrame(get('/metrics/curve')['points']).set_index('threshold')
        st.line_chart(curve[['precision', 'recall', 'f1']], height=320)
        st.caption('Precision rises and recall falls as the threshold goes up. '
                   'Every point is the same model, only the cut moves.')

    st.line_chart(curve[['cost']], height=200)
    st.caption(
        f"Total cost at {suggested['best_cost']['false_negative_cost']} per missed "
        f"customer and {suggested['best_cost']['false_positive_cost']} per wasted "
        f"offer. Its lowest point sits well below the one that maximises F1.")

# importance

with importance_tab:
    importance = get_importance()

    st.subheader('Permutation importance')
    st.caption('How much the ranking degrades when one column is shuffled. '
               'Measured with roc_auc, so these numbers do not move with the '
               'threshold.')
    permutation = pd.DataFrame(importance['permutation']).set_index('feature')
    st.bar_chart(permutation['mean'], horizontal=True, height=460)

    st.subheader('Model coefficients')
    st.caption('A second method. Logistic regression is additive, so each '
               'coefficient is that column\'s contribution in log odds. '
               'Positive pushes towards churn.')
    coefficients = pd.DataFrame(importance['coefficients']).head(15)
    st.dataframe(coefficients, width='stretch', hide_index=True)
