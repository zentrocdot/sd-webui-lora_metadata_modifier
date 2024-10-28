#!/usr/bin/python3
'''sd-webui-lora_metadata_writer

Extension for AUTOMATIC1111.

Version 0.0.0.1
'''
# pylint: disable=invalid-name
# pylint: disable=import-error
# pylint: disable=line-too-long
# pylint: disable=no-member
# pylint: disable=bare-except
# pylint: disable=broad-except
# pylint: disable=global-statement
# pylint: disable=global-variable-not-assigned
# pylint: disable=inconsistent-return-statements
# pylint: disable=useless-return

# Import the Python modules.
import os
import json
import io
import shutil
from typing import BinaryIO
from pathlib import Path
import gradio as gr
import modules.sd_models as models
import modules.shared
from modules.ui import create_refresh_button
from modules import script_callbacks

# Get LoRA path.
LORA_PATH = getattr(modules.shared.cmd_opts, "lora_dir", os.path.join(models.paths.models_path, "Lora"))

# Create a private global dictionary.
_lora_dict = {}

# Set private variable.
_SortDir = False

# ********************
# Function lora_scan()
# ********************
def lora_scan(lora_dir: str, ext: list) -> (list, list):
    '''File scan for LoRA models.'''
    global _lora_dict
    subdirs, files = [], []
    for fn in os.scandir(lora_dir):
        if fn.is_dir():
            subdirs.append(fn.path)
        if fn.is_file():
            if os.path.splitext(fn.name)[1].lower() in ext:
                _lora_dict[fn.name] = fn.path
    for dirs in list(subdirs):
        sd, fn = lora_scan(dirs, ext)
        subdirs.extend(sd)
        files.extend(fn)
    return subdirs, files

# ************************
# Function get_lora_list()
# ************************
def get_lora_list() -> list:
    '''Simple function for use with components.'''
    global _lora_dict
    _lora_dict = {}
    lora_list = []
    lora_scan(LORA_PATH, [".safetensors"])
    lora_list = list(_lora_dict.keys())
    lora_list.sort(reverse=_SortDir)
    return lora_list

# ---------------------------
# Function read_header_data()
# ---------------------------
def read_header_data(file: BinaryIO) -> dict:
    '''Read header data from a given file.

    Parameter:
        file:   file to read from as binary
        return: header data as dict from json
    '''
    # Read the header size saved as little endian from first 8 bytes of file.
    header_size_bytes = file.read(8)
    # Convert the read little endian bytes to an integer.
    header_size_integer = int.from_bytes(header_size_bytes, byteorder='little')
    # Read the header data.
    header_data = file.read(header_size_integer)
    # Create the JSON data.
    json_data = json.loads(header_data)
    # Return the JSON data as dict.
    return json_data

# ---------------------------
# Function read_header_data()
# ---------------------------
def read_metadata(file_name: str) -> dict:
    """Read file to extract the metadata.

    Parameter:
        file_name: name of the file to be read
        return:    metadata as dict from json
    """
    def type_json(value):
        '''Return type of JSON substructure.'''
        try:
            return type(json.loads(value))
        except:
            pass
    # Create an empty dictionary.
    metadata = {}
    # Open binary file for readonly reading.
    with open(file_name, 'rb') as file:
        # Get the header data from the file.
        header_data = read_header_data(file)
    # Extract the metadata from the header.
    for key, value in header_data.get("__metadata__", {}).items():
        # Get value to key from metadata.
        metadata[key] = value
        # Check if value is of type string and if string is dict.
        if isinstance(value, str) and type_json(value) == dict:
            # Try to get the value as JSON data.
            try:
                metadata[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    # Return the metadata as dict.
    return metadata

# -------------------------
# Function write_metadata()
# -------------------------
def write_metadata(old_file_name: str, new_file_name: str, metadata: dict):
    '''Creates a new .safetensor file from given and modified file content and
    writes it with the modified metadata into a new file.

    Parameter:
        old_file_name: name of the original file
        new_file_name: name of the modified file
        metadata:      metadata as dict
    '''
    # Initialise the return code.
    return_code = None
    # Catch errors.
    try:
        # Open a binary file for readonly reading.
        with open(old_file_name, 'rb') as old_file:
            # Extract the header data and the header size from the given file.
            old_header_data = read_header_data(old_file)
            # Overwrite the metadata in the header.
            old_header_data['__metadata__'] = metadata
            # Convert modified header data back to a binary string.
            new_header_data = json.dumps(old_header_data, separators=(',', ':')).encode('utf-8')
            # Open a new file for writing.
            with open(new_file_name, 'wb') as new_file:
                # Calculate the new offset value.
                offset = len(new_header_data)
                # Write the new offset value into the file.
                new_file.write(offset.to_bytes(8, 'little'))
                # Write the new header data into file.
                new_file.write(new_header_data)
                # Calculate chunk size based on buffer size for wrtiting the tensor to file.
                chunk_size = io.DEFAULT_BUFFER_SIZE
                # Read first chunk data for writing.
                chunk = old_file.read(chunk_size)
                # Write chunk data to file as long there is data.
                while chunk:
                    # Write chunk data to the file.
                    new_file.write(chunk)
                    # Read the next new chunk data.
                    chunk = old_file.read(chunk_size)
        # Set return code to 0 (success).
        return_code = 0
    except Exception as err:
        # Print error.
        print(err)
        # Set return code to 1 (error).
        return_code = 1
    # Return the return code.
    return return_code

# +++++++++++++++++++++
# Function on_ui_tabs()
# +++++++++++++++++++++
def on_ui_tabs():
    '''Method on_ui_tabs()'''
    def get_file_tag_name(fn):
        tag = "ss_output_name"
        basename = Path(fn).stem
        fp = _lora_dict.get(fn)
        metadata = read_metadata(fp)
        return basename
    # Create a new block.
    with gr.Blocks(analytics_enabled=False) as ui_component:
        # Create a new row.
        with gr.Row():
            input_file = gr.Dropdown(choices=get_lora_list(), label="LoRA File List")
            create_refresh_button(input_file, get_lora_list,
                                  lambda: {"choices": get_lora_list()},
                                  "metadata_utils_refresh_1")
            sort_fw_bw = gr.Radio(choices=["Forward", "Backward"], value="Forward",
                                  label="Sorting Direction", info="",
                                  scale=2, min_width=50)            
            def change_sort_fw_bw(rb_state):
                global _SortDir
                _SortDir = False
                if rb_state == "Forward":
                    _SortDir = False
                elif rb_state == "Backward":
                    _SortDir = True
                return []
            sort_fw_bw.change(change_sort_fw_bw, inputs=[sort_fw_bw], outputs=[])
        with gr.Row():
            filename = gr.Textbox(value="", lines=1, render=True,
                                  interactive=False, inputs=None, info="",
                                  label="Selected filename without extension")
            update_button = gr.Button(value="Update")
            input_file.input(fn=get_file_tag_name,
                             inputs=[input_file],
                             outputs=[filename])
        # Create a new row.
        with gr.Row():
            json_output = gr.Code(lines=10, label="Metadata as JSON", language="json", interactive=True)
            input_file.input(
                fn=read_lora_metadata,
                inputs=[input_file],
                outputs=[json_output]
            )
            def adjust_metadata(src, metadata):
                tag = Path(src).stem
                src_path = _lora_dict.get(src)
                dst_path = ''.join([src_path, ".bak"])
                shutil.copyfile(src_path, dst_path)
                print(metadata)
                write_metadata(dst_path, src_path, metadata)
                return []
            update_button.click(
                adjust_metadata,
                inputs=[input_file, json_output],
                outputs=[]
            )
            update_button.click(
                get_file_tag_name,
                inputs=[input_file],
                outputs=[filename]
            )
            update_button.click(
                read_lora_metadata,
                inputs=[input_file],
                outputs=[json_output]
            )
    return [(ui_component, "Write LoRA Metadata", "write_lora_metadata_tab")]

# Invoke a callback function.
script_callbacks.on_ui_tabs(on_ui_tabs)

# ++++++++++++++++++++++++
# Function get_lora_path()
# ++++++++++++++++++++++++
def get_lora_path(lora_file: str) -> str:
    '''Get the path to the LoRA file.'''
    if not isinstance(lora_file, str):
        lora_file = ""
    if not os.path.isfile(os.path.join(LORA_PATH, lora_file)):
        return ""
    return os.path.join(LORA_PATH, lora_file)

# +++++++++++++++++++++++++++++
# Function read_lora_metadata()
# +++++++++++++++++++++++++++++
def read_lora_metadata(input_file: str) -> json:
    '''Read the LoRA metadata.'''
    if selected_model := get_lora_path(_lora_dict.get(input_file)):
        if metadata := models.read_metadata_from_safetensors(selected_model):
            return json.dumps(metadata, indent=4, ensure_ascii=False)
        return "No metadata"
    return "No model"
