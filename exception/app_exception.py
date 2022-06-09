class AppException(Exception):

    def __init__(self, message='An app error has occurred'):
        super().__init__(message)
