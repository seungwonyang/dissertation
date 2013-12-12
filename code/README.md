
code folder
---------------
<strong>d3 folder</strong><br>
: It is a JavaScript library (http://d3js.org/) used along with the Xpantrac UI.

<strong>Xpantrac_UI.html</strong><br>
: It is a user interface for Xpantrac, which communicates with SOLR, used for the usability study.

<strong>custom_stops.txt</strong><br>
: It contains additional stopwords.

<strong>pos_tagger.py</strong><br>
: This script contains part-of-speech taggers that are chained as the backoff taggers of each other, in order to improve the tagging accuracy.

<strong>stopwords.txt</strong><br>
: It contains a list of general purpose stopwords.

<strong>Xpantrac_bing_buildCache.py</strong><br>
: This script expands provided input texts using a commercial search engine API (e.g., Bing API), then stores the expanded textual information in a database table for later analysis and topic extraction.

<strong>Xpantrac_bing_DB.py</strong><br>
: This script extracts the expanded textual information from a database table, which had been stored using Xpantrac_bing_buildCache.py.  One of the benefits is to extract topics for each varying parameter setting, for example, different values of the "number of API return" parameter.

<strong>Xpantrac_extractTopics_bing.py</strong><br>
: This script is an autonomous topic extraction script to identify topic tags given textual documents.  The Bing Azure API was used in this script.
