from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.results import get_results

project_name = "Karma Farmer"
net_id = "Brian Lu: bl694, Maria Silaban: djs488, Vivian Li: vml39, William Wang: wow7, Yuna Shin: ys457"

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		count = 0
		output_message = ''
	else:
		output_message = "Suggested Subreddits for: " + query
		data = []
		count = 0; 
		for i in get_results(query):
			#data.append("subreddit: " + i['subreddit'] + ",   " + "score: " + str(i['score']))
			data.append(i['subreddit'])
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)

