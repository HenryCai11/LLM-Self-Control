class AttributeUndefinedError(Exception):
    """Attribute Undefined Error Class"""
    def __init__(self, message="Attribute Undefined"):
        self.message = message
        super().__init__(self.message)