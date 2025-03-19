import sys 
import logging



def error_message_detail(error:Exception , error_detail:sys)->str:


    # extract trackback details ( Exception information )
    _, _, exc_tb = error_detail.exc_info()


    # get the file name where exception occured
    file_name = exc_tb.tb_frame.f_code.co_filename


    # create formated error messge string with filename , line number and actual error
    line_number = exc_tb.tb_lineno

    error_message = f"error occured in python script:[{file_name}]at number:[{line_number}]:str [{error}]"
    logging.error(error_message)
    return error_message

class MyException(Exception):
    "custom exception class while handling error in visa application"
    def __init__ (self , error_message:str , error_details:sys):
        

        # call the base class constructor with error message 
        super().__init__(error_message)
        
        self.error_meassage = error_message_detail( error_message , error_details)
    def __str__(self)-> str :
        return self.error_meassage
    





