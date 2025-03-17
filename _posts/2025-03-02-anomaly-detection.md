---
layout: post
title:  Anomaly Detection with Isolation Forests
date:   2025-03-02
categories: security
---
This notebook was created based on the very last project I did for my MBA degree. I was taking the class "Programming for Business Analytics" and the assignment was essentially to do a project that was relevant to our work and applied what we learned in the class. The final deliverable was an "Amazon 6-pager" and oddly the professor didn't want there to be any code in it. The class taught us basic SQL and Python with some tenuous applications to traditional statistical analytics so naturally I decided to do a machine learning project.


```python
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
```

### Anomaly Detection
By day I am a security "engineer". By night... I sleep. One of the many responsibilities I have in my day job is to respond to alerts that come in from our managed security service provider (MSSP). My organization has a very small security team and, because of my aforementioned night job, we need someone to keep an eye on things when I'm not awake. So, we send all of our log data to this MSSP and trust (pay) them to filter the signal from the noise. 

One of the security challenges my organization has is: what is the definition of an anomaly? We are an institution of higher education (a college not a dispensary) and as such our users often move about the world logging into things willy nilly from all sorts of devices. This makes it challenging to determine whether a login is truly anomalous and possibly malicious. Decent hackers these days make it even harder by using VPNs or public cloud to mask their true location. So the problem I wanted to gain some insight into was: how does one detect an anomaly, and can I do it "better" than Microsoft?

### The data
The first problem I had was finding data that I knew had anomalies in it. I have access to endless amounts of login data from my organization, but just like when someone asks you out of context what your favorite song is, I just couldn't find anything that worked. 

Ultimately, I doctored a data set since this was more of a proof of concept than a true attempt at developing something for production. I took my own login data and sprinkled in some random logins from other users. 

The next callenge was deciding what variables to focus on. I had 17 categories for each login attempt. 


```python
df = pd.read_csv("/Users/jbackon/Repos/blog/jrbackon.github.io/backon_data.csv")
print(len(df.columns))
df.columns
```

    17





    Index(['Date (UTC)', 'User agent', 'Username', 'Application', 'Resource',
           'IP address', 'Location', 'Status', 'Sign-in error code',
           'Failure reason', 'Client app', 'Browser', 'Operating System',
           'Multifactor authentication result', 'IP address (seen by resource)',
           'Conditional Access', 'Managed Identity type'],
          dtype='object')



Usually what I look at when examining logs is: location, device, and whether the login was successsful. Based on this, I decided to use four variables to help detect anomalies:
- Status (whether the login was successful or not)
- User Agent (information sent with the login that has clues to the type of device used to login)
- Browser (more information about the device used to login)
- Location

Now I just needed to figure out how to do the detection.

### Isolation Forests
After doing some research it turns out that there is a very clever algorithm to detect outliers in a given range of numbers. It's called an isolation forest. Here's how it works.

Say I have a set of data (here represented by a list of numbers).


```python
data = [1,6,9,14,20,23,46,1002]
```

To isolate the outlier what I can do is pick a random number in the range of values in the set. In this case I'll pick 15. I then split the data into two groups. Numbers smaller than 15 and numbers bigger.


```python
branch1 = [1,6,9,14]
branch2 = [20,23,46,1002]
```

I then repeat this process until all numbers have been isolated, or I reach a set number of splits. We can see if I pick 7 for the smaller set and 50 for the bigger that I will have already isolated one number. The full set of splits is known as a "tree".


```python
branch3 = [1,6]
branch4 = [9, 14]
branch5 = [20,23,46]
leaf1 = [1002]
```

Once a branch only has one value then it becomes a leaf. The values that form leafs the fastest are most likely the outliers. The algorithm repeats this process over and over again using different numbers to do the splitting. This creates a group of "trees" or a "forest" hence the name of the algorithm. 

The machine learning aspect has to do with the relative "distance" to a leaf for each data point across all of the trees. Based on these combined distances the algorithm assigns an "anomaly score" that suggests the probability that the point is an outlier. Then, based on inputs to the algorithm it picks a threshold value for the distribution of anomaly scores and all values smaller than the threshold get classified as anomalies.

It might sound a little complicated, and I don't know a good way of doing visualizations in python (yet), but I have an image later that might help. The nice things about this algorithm though is it is super fast and doesn't take much compute power. This is useful especially if this method is going to be used for real-time detection.

### Preparing the data
So, I had my algorithm, but there was a problem. Most of my data was categorical and not numerical. I needed numerical data to use the isolation forest algorithm. Luckily, Pandas has a neat method for encoding data. This simply assigns a random number to each value in each category.


```python
df['status_encoded'] = df['Status'].astype('category').cat.codes
df['user_agent_encoded'] = df['User agent'].astype('category').cat.codes
df['browser_encoded'] = df['Browser'].astype('category').cat.codes
df['location_encoded'] = df['Location'].astype('category').cat.codes
```

So now instead of:


```python
non_encoded = ['Status', 'User agent', 'Browser'] # location removed for privacy
df[non_encoded].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Status</th>
      <th>User agent</th>
      <th>Browser</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Success</td>
      <td>Mozilla/5.0 (Windows NT 10.0; Win64; x64) Appl...</td>
      <td>Edge 131.0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Interrupted</td>
      <td>Mozilla/5.0 (Windows NT 10.0; Win64; x64) Appl...</td>
      <td>Edge 131.0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Success</td>
      <td>Windows-AzureAD-Authentication-Provider/1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Success</td>
      <td>Mozilla/4.0 (compatible; MSIE 7.0; Windows NT ...</td>
      <td>IE 7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Success</td>
      <td>Mozilla/5.0 (Windows NT 10.0; Win64; x64) Appl...</td>
      <td>Edge 131.0.0</td>
    </tr>
  </tbody>
</table>
</div>



My data looks like:


```python
test_columns = ['user_agent_encoded', 'browser_encoded', 'location_encoded', 'status_encoded']
df[test_columns].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_agent_encoded</th>
      <th>browser_encoded</th>
      <th>location_encoded</th>
      <th>status_encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>-1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>2</td>
      <td>8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



I'm almost ready to go. I now need to split the data into test and prediction sets as is standard for any machine learning problem. I chose to use a 60:40 test:prediction split for reasons...


```python
test_data, prediction_data = train_test_split(df, test_size=0.6, random_state=42)

```

### Training and using the model
I can now train the IsolationForest model provided by sklearn. 
- **n_estimators**: provides the maximum number of branches for each tree.
- **contamination**: is my best guess as to how much of the data is an anomaly. This is used to help the algorithm pick an appropriate anomaly score threshold.

I fit the model with the test data only using the four encoded variables I mentioned earlier.


```python
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(test_data[test_columns])
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>IsolationForest(contamination=0.01, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>IsolationForest</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.IsolationForest.html">?<span>Documentation for IsolationForest</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>IsolationForest(contamination=0.01, random_state=42)</pre></div> </div></div></div></div>



Now, that the model is trained I apply it to the prediction set to see how it performs. In the code below, I'm adding the anomaly score and determination of whether the data point is an anomaly to the prediction data set just so I can review the results based on the original data.


```python
prediction_data['anomaly_score'] = model.decision_function(prediction_data[test_columns])
prediction_data['is_anomalous'] = model.predict(prediction_data[test_columns])  # -1: anomaly, 1: normal
```

Here is an example of some encoded data and the associated anomaly score. 


```python
prediction_data[test_columns + ['anomaly_score', 'is_anomalous']].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_agent_encoded</th>
      <th>browser_encoded</th>
      <th>location_encoded</th>
      <th>status_encoded</th>
      <th>anomaly_score</th>
      <th>is_anomalous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>428</th>
      <td>13</td>
      <td>-1</td>
      <td>8</td>
      <td>2</td>
      <td>0.227029</td>
      <td>1</td>
    </tr>
    <tr>
      <th>440</th>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0.292787</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>-0.014148</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>468</th>
      <td>8</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>0.312700</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>9</td>
      <td>2</td>
      <td>8</td>
      <td>2</td>
      <td>0.261850</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



To better understand when a given anomaly score is classified as an anomaly it helps to look at the distribution of anomaly scores shown below.


```python
# Histogram of anomaly scores
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.histplot(prediction_data['anomaly_score'], kde=True, bins=30, color='purple')
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.axvline(x=0, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.show()

```


    
![png](https://github.com/jrbackon/jrbackon.github.io/blob/main/project_datasplit_files/project_datasplit_28_0.png?raw=true)
    


The histogram shows that anything with an anomaly score less than 0 was classified an an anomaly. The interesting thing to note is that, if I changed the **contamination** value when I was training the model, then the threshold would be different. In fact, the first time I built this model I used a value of 0.1 instead of 0.01. This told the model to expect 10% of the data to be anomalous. Given I only had about 50 data points in my set this value generated quite a few false positives. 

### Conclusions
Enought of the nuts and bolts! Did it work? This was fairly simple to discern by looking at the prediction set and checking the rows that had "-1" in the "is_anomalous" column. It turns out that, for the data in this paricular training/prediction split, there was only one anomaly. The prediction set had one row labeled with a "-1" in the "is_anomalous" column, and it was indeed the one anomalous login in the set. So, the model worked.

What is harder to discern from this simple experiment is: which variables were most effective at determining the anomaly? Or, how many variables are required? If I used more variables to train the model would it be more or less effective? When I first did this project, I tried visualizing the data in different ways. I tried plotting two variables at a time, I tried a 3D plot, but ultimately I was working in a 4D space and there is no easy way to visualize the interaction of 4 variables on a 2D screen. I suspect there are way to do this, but I will need to learn more about it.

So my ultimate conclusion is that I definitely can't do this better than Microsoft. Building a model is one thing. Building a production ready web application that can ingest data for thousands of users and provide near real-time anomaly detections is quite another. I like that I now better understand how the underlying technologies work and feel that I have a good foundation for more advanced study in this area. Specifically, it might be worth digging into the algorithm itself and seeing how it works under the hood. If I do this, I'll be sure to write about it.
