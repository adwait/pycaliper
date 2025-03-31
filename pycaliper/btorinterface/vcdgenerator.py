"""
    This script reads a dump file (or dump string) with lines of the form:

    <line_number> <value> <signal_name>

    For example:
    121 001 state_1
    ...
    1324 010 state_2

    Interpretation:
    • The first token is simply a line number and is ignored.
    • The third token is of the form "baseName_cycle". The “cycle” (last underscore-delimited token)
        is converted to an integer and used as the VCD simulation time. The rest (baseName) is used as
        the signal name.

    Each signal's width is inferred from the length of the “value” string.
"""


import string
from collections import OrderedDict

from btor2ex import Assignment, BTORModel


def unique_id_generator():
    """
    Generates a sequence of unique short identifiers to be used as VCD ids.
    It first yields single printable characters and then combinations of increasing length.
    """
    printable = string.printable.strip()  # remove whitespace characters
    # Yield one-character ids.
    for c in printable:
        yield c
    length = 2
    while True:

        def rec_gen(prefix, length):
            if length == 0:
                yield prefix
            else:
                for c in printable:
                    yield from rec_gen(prefix + c, length - 1)

        for identifier in rec_gen("", length):
            yield identifier
        length += 1


def process_signal_and_time(full_signal):
    """
    Process a signal name from the dump.
    The full signal is expected to be of the form:
       baseName_cycle
    where the last underscore-delimited token (cycle) is used as the simulation time.
    Returns a tuple (base_signal, cycle) where:
       - base_signal is the signal name without the cycle suffix.
       - cycle is the simulation time (as an integer).
    If no underscore is found, the entire string is the base name and cycle is 0.
    """
    if "_" in full_signal:
        parts = full_signal.rsplit("_", 1)
        try:
            cycle = int(parts[1]) - 1
        except ValueError:
            # If conversion fails, assume cycle 0.
            cycle = 0
        return parts[0], cycle
    return full_signal, 0


def parse_lines(lines):
    """
    Processes an iterable of lines (from a file or a string) and returns:
       - signals: an OrderedDict mapping base signal names -> dict with keys:
             'width': inferred width (int)
             'vcd_id': placeholder for later assignment.
       - events: a list of (time, signal, value) tuples, where time is taken from the signal name suffix.

    It assumes each line is of the form:
         <line_number> <value> <full_signal>
    where only the <value> and the split result of <full_signal> (base name and cycle) are used.
    """
    signals = OrderedDict()
    assignments: dict[int, Assignment] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("..."):
            continue
        parts = line.split()
        # Ensure there are at least 3 tokens.
        if len(parts) < 3:
            continue
        # The first token is a line number (ignored).
        value = parts[1]
        full_signal = parts[2]
        base_signal, cycle = process_signal_and_time(full_signal)
        width = len(value)
        if base_signal in signals:
            # Update width if a later value uses more bits.
            if signals[base_signal] < width:
                signals[base_signal] = width
        else:
            signals[base_signal] = width
        if cycle not in assignments:
            assignments[cycle] = {}
        assignments[cycle][base_signal] = int(value, 2)
    return signals, assignments


def events_from_assignments(assignments: dict[int, Assignment]):
    events = []
    for cycle, assignment in assignments.items():
        for signal, value in assignment.items():
            events.append((cycle, signal, value))
    return events


def parse_dump_file(filename: str):
    """
    Reads the dump file given by filename and parses its content.
    Returns a tuple (signals, events).
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    return parse_lines(lines)


def parse_dump_string(input_str: str):
    """
    Parses dump content provided as a string.
    Returns a tuple (signals, events).
    """
    lines = input_str.splitlines()
    return parse_lines(lines)


def write_vcd(
    signals: dict[str, int], assignments: dict[int, Assignment], timescale="1ns"
):
    """
    Generate a VCD trace from the given signals and events.
    'signals' is a dict mapping signal names -> width.
    'events' is a list of tuples (time, signal, value).

    The VCD output uses simulation time stamps (cycles) derived from the signal name suffix.
    """
    uid_gen = unique_id_generator()
    signal_uids = {}
    # Assign unique VCD ids to each signal.
    for sig in signals:
        signal_uids[sig] = next(uid_gen)

    events = events_from_assignments(assignments)

    out_lines = []
    # Header.
    out_lines.append("$date")
    out_lines.append("   Generated by dump_to_vcd.py")
    out_lines.append("$end")
    out_lines.append("$version")
    out_lines.append("   Python VCD generator")
    out_lines.append("$end")
    out_lines.append("$timescale {} $end".format(timescale))
    out_lines.append("$scope module top $end")
    for signal, width in signals.items():
        sig_id = signal_uids[signal]
        # VCD variable declaration.
        out_lines.append("$var wire {} {} {} $end".format(width, sig_id, signal))
    out_lines.append("$upscope $end")
    out_lines.append("$enddefinitions $end")
    out_lines.append("$dumpvars")
    # Initial dump for each signal set to unknown ('x').
    for signal, width in signals.items():
        sig_id = signal_uids[signal]
        if width == 1:
            out_lines.append("x{}".format(sig_id))
        else:
            out_lines.append("b{} {}".format("x" * width, sig_id))
    out_lines.append("$end")

    # Group events by simulation time.
    time_events = OrderedDict()
    for time, signal, value in events:
        if time not in time_events:
            time_events[time] = []
        time_events[time].append((signal, value))

    # Write out events sorted by simulation time.
    for t in sorted(time_events.keys()):
        out_lines.append("#{}".format(t))
        for signal, value in time_events[t]:
            sig_id = signal_uids[signal]
            width = signals[signal]
            if width == 1:
                out_lines.append("{}{}".format(value, sig_id))
            else:
                out_lines.append("b{} {}".format(value, sig_id))

    out_lines.append("#{}".format(t + 1))

    return "\n".join(out_lines)


def convert_to_vcd_from_btorstr(model: str) -> str:
    """
    Convert a BTOR model string to a VCD trace.
    """
    btor_model = parse_dump_string(model)
    return write_vcd(btor_model.signals, btor_model.assignments)


def convert_to_vcd_from_btorfile(filename: str) -> str:
    """
    Convert a BTOR model file to a VCD trace.
    """
    signals, assignments = parse_dump_file(filename)
    return write_vcd(signals, assignments)


class BTORModelParser:
    def __init__(self):
        pass

    def parse(self, model: str) -> BTORModel:
        """
        Parse the BTOR model and return the signals and events.
        """
        signals, assignments = parse_dump_string(model)
        return BTORModel(signals, assignments)
