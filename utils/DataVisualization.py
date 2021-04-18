
import numpy as np

def DisplaySlices(data, ColourMax):
    """
    Volume Slices visualization:
        - Left Atria mask
        - MRI_volumes slices

    Input:
    :param data: numpy array with the mask or MRI_volumes information
    :param ColourMax: data.max()

    Return: opens a page with the slices displayed
    """
    r = data.shape[0]
    c =data.shape[1]

    # DISPLAY ALL SLICES:
    import plotly.graph_objects as go

    nb_frames = data.shape[-1]
    volume = data
    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=((data.shape[-1]-1) - k) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[:,:,(data.shape[-1]-1) - k]),
        cmin=0, cmax=ColourMax
        ),
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=(data.shape[-1]-1) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[:,:,(data.shape[-1]-1)]),
        colorscale='Gray',
        cmin=0, cmax=ColourMax,
        colorbar=dict(thickness=20, ticklen=4)
        ))


    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
             title='Slices in volumetric data',
             width=600,
             height=600,
             scene=dict(
                        zaxis=dict(range=[0, (data.shape[-1]-1)], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
             updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )

    fig.show()


