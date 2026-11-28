"""Standalone tools that are not part of the runtime application.

Nothing under ``tools`` is imported by the framework or by ``app``.  These are
programs a person runs by hand, so they may depend on the firmware or on a
desktop GUI; moving them here keeps that freedom out of the shipped package.
"""
