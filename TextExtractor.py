import glob
import os

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox
from pdfminer.layout import LTTextLine


class TextExtractor:
    __file_path = None
    __directory_path = None
    # Номер страниц оглавления
    __pages_with_chapter_num = []
    # Страницы, где находится оглавление
    __pages_with_chapter = []
    __pages = []
    max_count_file = 1

    def __init__(self, directory_path=None, max_count_file=1, file_path=None):
        """
        :param directory_path: Path to directory. ex E:\OneDrive\НИКОР 2019 - 2020\2021\ВКР\pdf
        :param max_count_file: Maximum count of files
        """
        self.__directory_path = directory_path if directory_path else self.__directory_path
        self.__file_path = file_path if file_path else self.__directory_path
        self.max_count_file = max_count_file

    # for multiply files
    # def getFilesFromDirectory(self):
    #     """ Get recursive all pdf files directory """
    #     d = {}
    #     result = []
    #     for x in os.walk(self.directory_path):
    #         print("path is: ", x[1])
    #         print("path is: ", x[0])
    #         for y in glob.glob(os.path.join(x[0], '*.pdf')):
    #             result.append(y)
    #     print(result)
    #     print(d)
    #     return result
    def __cleartext(self, string):
        string = string.strip()
        return string

    def getChapters(self, file_path=None):
        # Путь к файлу
        file_path = file_path if file_path else self.__file_path
        # Получение страниц pdf-rb (генератор)
        pages = extract_pages(file_path)
        # Объекты страниц
        pages_with_chapter = []
        pages_with_chapter_num = []
        self.__document_pages_count = 0

        for page_layout in pages:

            self.__document_pages_count += 1
            for element in page_layout:  # LTPage
                self.__pages.append(page_layout)
                if isinstance(element, LTTextBox):
                    # Построчный обход каждой страницы
                    if 'оглавление' in element.get_text().lower() or 'заключение' in element.get_text().lower():
                        pages_with_chapter.append(page_layout)
                        pages_with_chapter_num.append(self.__document_pages_count)
        i = 0
        while i < len(pages_with_chapter_num):
            if pages_with_chapter_num[i] > 20:
                del pages_with_chapter_num[i]
            i += 1
        self.__pages_with_chapter_num = list(range(pages_with_chapter_num[0], pages_with_chapter_num[-1] + 1))
        self.__pages_with_chapter = extract_pages(file_path, page_numbers=pages_with_chapter_num)

        return {"pages_with_chapter_num": self.__pages_with_chapter_num,
                "pages_with_chapter": self.__pages_with_chapter, "pages": self.__pages}

    def getFileChapters(self) -> list:

        chapters = []

        tmp_text = ""
        for page_layout in self.__pages_with_chapter:

            for row in page_layout:
                if isinstance(row, LTTextBox):
                    s = row.get_text()
                    # Если название главы растиянуто на две строчки
                    if s.find("..") == -1:
                        tmp_text = self.__cleartext(s[:s.find("..")])
                        continue

                    text = self.__cleartext(s[:s.find("..")]) + tmp_text
                    page = self.__cleartext(''.join(i for i in s if i.isdigit()))[-2:]
                    print("textstrip = ", text)
                    print("pagestrip = ", page)

                    # Если текст не число, а второй символ с конца - число
                    # И есть номер
                    if not text.isdigit() and text and page:  # and text[-2:].isdigit():  # проверяем не пустые ли значения
                        chapters.append([page, text])
                        tmp_text = ""

                    # print(s.rfind("."))

                    # print(page)

        # chapters.pop(0)  # удаляем нулевой элемент
        print(self.__pages_with_chapter_num, "<++++++++++++++----------------------------------")
        print(chapters)
        return chapters

    def generatePagesForChapters(self, chapters: list) -> list:
        """
        Generate pages list for each chapter
        :param chapters: List of chapter with pages
        ex. [['8', 'Введение '], ['11', 'Термины, определения и сокращения ']]
        :return: List of chapter with page begin and end
        ex. [['8', 'Введение ', [8, 9, 10]], ['11', 'Термины, определения и сокращения ', [11, 12]]]
        """
        print(chapters)
        for i, value in enumerate(chapters):
            nextP = i + 1
            try:
                d = list(range(int(value[0]), int(chapters[nextP][0]), 1))
                chapters[i].append(d)
            except (IndexError, ValueError) as e:
                print("Catch a error", e)
                print("value\n --------------------------------------------\n")
                del chapters[i]
                # d = list(range(int(value[0]), int(self.__document_pages_count), 1))
                # chapters[i].append(d)
        return chapters

    def getContent(self, chapters):
        '''
        Get content of chapters
        :param chapters:
        :return:
        '''
        chapters_with_content = []
        for i, value in enumerate(chapters):
            value[2] = [x - 1 for x in value[2]]  # Так как массив страниц с 0 начинается
            data = extract_pages(self.__file_path, page_numbers=value[2])
            if value[2]:
                for page_layout in data:
                    chapter_text = ""
                    for page in page_layout:
                        if isinstance(page, LTTextBox) or isinstance(page, LTTextLine):
                            chapter_text = chapter_text + page.get_text()
                    chapters_with_content.append([value[1], chapter_text])
                    print([value[1], chapter_text])
        return chapters_with_content
