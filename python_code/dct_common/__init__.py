"""Shared code for the JPEG-DCT MLP and CNN classifiers.

Both networks read features straight out of the JPEG DCT bitstream (never
decoding to pixels) for the same openimages_8class-based dataset.
Everything that isn't specific to one model's architecture lives here:
config, dataset build, feature extraction, quantization, metrics, and
device helpers.
"""
