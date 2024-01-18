import streamlit as st


class CommonHelper:
    @staticmethod
    def add_para(text: str = "", alignment: str = "justify", unsafe: bool = True):
        """
        Adds a paragraph to the Streamlit page, with customization such as alignment and unsafe HTML options.
        :param text: A string value: the text paragraph to be inputted
        :param alignment: A string value: only accepts left, right, center, and justify alignment.
        :param unsafe: A boolean value: declares if unsafe HTML formats should be used.
        :return: A null value: generates the paragraph catered to customization.
        """
        if text:
            return st.markdown(f'<p style="text-align:{alignment};">{text}</p>', unsafe_allow_html=unsafe)
        else:
            raise Exception("Missing text when using add_para method.")

    @staticmethod
    def add_link(text: str = "", link: str = "", target: str = "_blank"):
        """
        Adds a link to the text, with customization such as link and target.
        :param text: A string value: the words to be used that upon click, redirects to the link.
        :param link: A string value: the URL of the link within the text hyperlink.
        :param target: A string value: describes the target, otherwise left as _blank.
        :return: A null value: generates the hyperlink catered to customization.
        """
        if text and link:
            return f'<a href="{link}" target="{target}">{text}</a>'
        else:
            raise Exception("Missing text and/or link when using add_link method.")

    @staticmethod
    def add_list(entries: list = None, numbered: bool = False, unsafe: bool = True):
        """
        Adds an ordered / unordered list, with customization such as numbered and unsafe HTML options.
        :param entries: A list: Elements to be displayed in a list should reside within a list data structure.
        :param numbered: A boolean value: True if ordered (numbered), False if unordered (dotted).
        :param unsafe: A boolean value: declares if unsafe HTML formats should be used.
        :return: A null value: generates an ordered / unordered list catered to customization.
        """
        if entries:
            if numbered:
                for number, element in enumerate(entries):
                    st.markdown(f"{number + 1}. {element}", unsafe_allow_html=unsafe)
            else:
                for element in entries:
                    st.markdown(f"* {element}", unsafe_allow_html=unsafe)
        else:
            raise Exception("Missing entries list when using add_list method.")
