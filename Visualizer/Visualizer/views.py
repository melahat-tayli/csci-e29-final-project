import os
from os.path import join

import numpy as np
import pandas as pd
from bokeh.document import Document
from bokeh.embed import server_document
from bokeh.layouts import column, row
from bokeh.models import Select
from bokeh.plotting import figure
from bokeh.themes import Theme
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

theme = Theme(filename=join(settings.THEMES_DIR, "theme.yaml"))


def crossfilter_handler(doc: Document) -> None:
    df = pd.read_csv(os.path.join(os.getcwd(), "../data/heart.csv")).copy()

    columns = sorted(df.columns)
    discrete = [x for x in columns if df[x].dtype == object]
    continuous = [x for x in columns if x not in discrete]

    def create_figure():
        xs = df[x.value].values
        ys = df[y.value].values
        x_title = x.value.title()
        y_title = y.value.title()

        kw = dict()
        if x.value in discrete:
            kw["x_range"] = sorted(set(xs))
        if y.value in discrete:
            kw["y_range"] = sorted(set(ys))
        kw["title"] = "%s vs %s" % (x_title, y_title)

        p = figure(
            plot_height=600, plot_width=800, tools="pan,box_zoom,hover,reset", **kw
        )
        p.xaxis.axis_label = x_title
        p.yaxis.axis_label = y_title

        if x.value in discrete:
            p.xaxis.major_label_orientation = pd.np.pi / 4

        hist, edges = np.histogram(xs, density=True, bins=50)
        p = figure(title=x_title, tools="", background_fill_color="#fafafa")
        p.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color="navy",
            line_color="white",
            alpha=0.5,
        )

        p.y_range.start = 0
        p.legend.location = "center_right"
        p.legend.background_fill_color = "#fefefe"
        p.xaxis.axis_label = "x"
        p.yaxis.axis_label = "Pr(x)"
        p.grid.grid_line_color = "white"
        return p

    def callback(attr: str, *args, **kwargs) -> None:
        layout.children[1] = create_figure()

    x = Select(title="X-Axis", value="sex", options=columns)
    x.on_change("value", callback)

    y = Select(title="Y-Axis", value="age", options=columns)
    y.on_change("value", callback)

    size = Select(title="Size", value="None", options=["None"] + continuous)
    size.on_change("value", callback)

    color = Select(title="Color", value="None", options=["None"] + continuous)
    color.on_change("value", callback)

    controls = column(x, y, color, size, width=200)
    layout = row(controls, create_figure())

    doc.theme = theme
    doc.add_root(layout)
    doc.title = "Crossfilter"


def crossfilter(request: HttpRequest) -> HttpResponse:
    script = server_document(request.build_absolute_uri())
    return render(request, "embed.html", dict(script=script))
