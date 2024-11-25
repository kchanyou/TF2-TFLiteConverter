import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Folder that the saved model is located in',
                    default='C:/venv/tf2/models/research/object_detection/exported_models/my_saved_model/saved_model')
parser.add_argument('--output', help='Folder that the tflite model will be written to',
                    default='C:/venv/tf2/models/research/object_detection/exported_models/my_tflite_model')
args = parser.parse_args()

output_dir = args.output
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(args.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

output_path = os.path.join(output_dir, 'model.tflite')

try:
    with tf.io.gfile.GFile(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model successfully saved to {output_path}")
except Exception as e:
    print(f"Error saving model: {e}")
