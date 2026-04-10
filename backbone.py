# https://huggingface.co/ <- github for ai research
# https://github.com/moment-timeseries-foundation-model/moment <- backbone source
# Notes from "C:\Users\GSanders2\OneDrive - Hill & Smith PLC\Masters Drive\Capstone\Research\Paper downloads\Moment a family of open time-series foundation models.pdf"

'''
We address variable length by restricting MOMENT’s
input to a univariate time series of a fixed length
T = 512. As is common practice, we sub-sample longer
time series, and pad shorter ones with zeros on the left.
Moreover, segmenting time series into patches quadratically
reduces MOMENT’s memory footprint and computational
complexity, and linearly increases the length of time series
it can take as input. 

.to("xpu") #train on graphics card
'''

from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={
        "task_name": "forecasting",
        "forecast_horizon": 96
    },
)
model.init()