from bs4 import BeautifulSoup, Tag
import htmlmin

MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100


def remove_hyperlinks(soup):
    # Remove all hyperlinks
    for a in soup.findAll('a'):
        a.replaceWithChildren()

    return soup


def cleanup_table_html(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Remove unnecessary tags
    for tag in soup.find_all(['caption', 'p', 'a', 'code']):
        tag.replace_with(tag.get_text())

    # Remove unnecessary attributes but preserve 'colspan'
    for tag in soup(True):
        if 'colspan' in tag.attrs:
            tag.attrs = {'colspan': tag['colspan']}
        else:
            tag.attrs = {}

    # Convert HTML back to string and remove unnecessary whitespaces
    cleaned_html = str(soup)

    # Minify the HTML
    cleaned_html = cleaned_html.replace("\n", "").replace("\r", "")
    cleaned_html = ' '.join(cleaned_html.split())

    # Check if the cleaned HTML length exceeds the maximum section length
    rows = soup.find_all('tr')
    tables = []
    current_table = BeautifulSoup('<table></table>', 'html.parser').table
    current_length = 0

    for row in rows:
        row_length = len(str(row))
        if current_length + row_length > MAX_SECTION_LENGTH:
            # If adding the next row will exceed the limit, start a new table
            tables.append(str(current_table))
            current_table = BeautifulSoup('<table></table>', 'html.parser').table
            current_length = 0
        current_table.append(row)
        current_length += row_length

    # Append the last table if it's not empty
    if len(current_table.find_all('tr')) > 0:
        tables.append(str(current_table))

    # Return all tables as a single string
    return ''.join(tables)


# Define the function to extract HTML content and preserve sentence structure
def extract_html_content(soup):
    text_parts = []
    for element in soup.recursiveChildGenerator():
        if isinstance(element, str):
            # Remove leading/trailing whitespaces and replace multiple spaces with single space
            text = ' '.join(element.strip().split())
            if text:  # if the text is not empty
                # Check if the previous element was also text
                if text_parts and not text_parts[-1].endswith('\n'):
                    # If the previous element did not end with a newline,
                    # this means the current text is a continuation of the previous line
                    text_parts[-1] += ' ' + text
                else:
                    # Otherwise, this is a new line
                    text_parts.append(text)
        elif isinstance(element, Tag) and element.name in ['p', 'br']:
            # Add a newline for each paragraph or line break
            text_parts.append('\n')

    return ''.join(text_parts)


# Define the function to insert tables at markers
def insert_tables(text, table_texts):
    # Sort the table texts by position in descending order
    table_texts.sort(key=lambda x: x[1], reverse=True)

    # Insert each table at its corresponding marker
    for table_text, position in table_texts:
        marker = f"TABLE_MARKER_{position}"
        text = text.replace(marker, table_text)

    return text


def get_document_text(filename):
    offset = 0
    page_map = []

    with open(filename, 'r', encoding='UTF-8') as file:
        html_content = file.read()

        # Make the HTML code more compact
        html_content = htmlmin.minify(html_content, remove_empty_space=True)

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        content_div = soup.find('div', class_='wh_topic_content')

        if content_div is not None:
            soup = BeautifulSoup(str(content_div), 'html.parser')
            # soup = encode_images_in_html(soup, filename)
            soup = remove_hyperlinks(soup)

            # Extract tables and replace them with markers in the soup
            table_texts = []
            for i, table in enumerate(soup.find_all('table')):
                table_text = str(table)

                # # Read HTML table into pandas DataFrame
                # dfs = pd.read_html(table_text, header=[0], flavor='bs4')
                # df = dfs[0]  # Let's assume that there's only one table in the HTML
                #
                # # Convert DataFrame back to HTML
                # html_table_clean = df.to_html(index=False, border=0, classes=None)
                #
                # # Minify the cleaned HTML table
                # html_table_clean_min = htmlmin.minify(html_table_clean, remove_empty_space=True)
                html_table_clean_min = cleanup_table_html(table_text)

                position = len(''.join(extract_html_content(soup).split('\n')))
                table_texts.append((html_table_clean_min, position))
                marker = BeautifulSoup(f"TABLE_MARKER_{position}", 'html.parser')
                table.replace_with(marker)

            # Extract remaining text
            text = extract_html_content(soup)

            # Insert tables at the markers
            text = insert_tables(text, table_texts)

            page_map.append((1, offset, text))
            offset += len(text)

    return page_map


def split_text_html(page_map):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]

    # Assuming that the page_map contains only one page with all HTML string
    html_string = page_map[0][2]

    length = len(html_string)
    start = 0
    end = length
    chunk_index = 0

    # Create a list to store the results
    results = []

    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and html_string[end] not in SENTENCE_ENDINGS:
                if html_string[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and html_string[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word

        if end < length:
            end += 1

        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and html_string[
            start] not in SENTENCE_ENDINGS:
            if html_string[start] in WORDS_BREAKS:
                last_word = start
            elif start > 6 and html_string[start - 7:start + 1] == "</table>":  # Check for the closing table tag
                break  # Stop searching
            start -= 1
        if html_string[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word

        if start > 0:
            start += 1

        section_text = html_string[start:end]
        # yield (section_text, chunk_index)
        # Append results instead of yielding for debugging convenience
        results.append((section_text, chunk_index))

        last_table_start = section_text.rfind("<table")
        if last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table"):
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP

        chunk_index += 1

    if start + SECTION_OVERLAP < end:
        # yield (html_string[start:end], chunk_index)
        # Append results instead of yielding for debugging convenience
        results.append((html_string[start:end], chunk_index))

        # Return the results list
    return results
