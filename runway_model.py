# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =========================================================================

# This example contains the minimum specifications and requirements
# to port a machine learning model to Runway.

# For more instructions on how to port a model to Runway, see the Runway Model
# SDK docs at https://sdk.runwayml.com

# RUNWAY
# www.runwayml.com
# hello@runwayml.com

# =========================================================================

# Import the Runway SDK. Please install it first with
# `pip install runway-python`.
import runway
from runway.data_types import number, text, image,vector,array
from example_model import ExampleModel
from detect import detect

# Setup the model, initialize weights, set the configs of the model, etc.
# Every model will have a different set of configurations and requirements.
# Check https://docs.runwayapp.ai/#/python-sdk to see a complete list of
# supported configs. The setup function should return the model ready to be
# used.
'''
setup_options = {
    'truncation': number(min=1, max=10, step=1, default=5, description='Example input.'),
    'seed': number(min=0, max=1000000, description='A seed used to initialize the model.')
}
'''
@runway.setup
def setup():
    return ExampleModel()


@runway.command(name='generate',
                inputs={ 'photo': image() },
                outputs={ 'ret':vector(length = 0),'images':image()  },
                description='Generates a red square when the input text input is "red".')
def generate(model, input):
    output_image = model.run_on_input(input)
    return {
        'ret': output_image,
        'images':images
    }

if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8000)

