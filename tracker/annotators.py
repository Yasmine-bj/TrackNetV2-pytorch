from supervision import Color, BoxAnnotator, LabelAnnotator, EllipseAnnotator, BoxCornerAnnotator

def create_annotators(color=(0, 0, 0)):
    col = Color(*color)
    return (
        BoxAnnotator(color=col, thickness=2),
        LabelAnnotator(color=col),
        EllipseAnnotator(),
        BoxCornerAnnotator(),
    )
