"""Format and stream foundation: readers, pixel buffers, color, audio,
images, and the ffmpeg compatibility encoder.

Foundation endpoints are not processors (planning 05): this package is
the ground a processor chain runs on. Modules here never import from
processor families, pipeline, or CLI code.
"""
