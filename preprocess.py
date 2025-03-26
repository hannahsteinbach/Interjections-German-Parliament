import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import itertools

# refer to stammdaten to get gender
stammdaten = "MdB-Stammdaten/MDB_STAMMDATEN.XML"
sd_tree = ET.parse(stammdaten)
sd_root = sd_tree.getroot()

def get_gender_from_name(full_name, root):
    '''Function to retrieve the gender of an MP from the official MDB Stammdaten'''

    def normalize_name(name):
        titles_to_remove = ["Dr.", "Dr. h. c.", "Prof.", "h. c."]
        for title in titles_to_remove:
            name = name.replace(title, "")
        return name.strip()

    # Normalize the full name
    normalized_full_name = normalize_name(full_name)

    # Split the full name into first name and last name (assuming the last part is the last name)
    names = normalized_full_name.split()
    first_name = " ".join(names[:-1]) if len(names) > 1 else ""
    primary_first_name = names[0] if names else ""
    secondary_first_name = names[1] if len(names) > 1 else ""
    last_name = names[-1] if names else ""

    for mdb in root.findall('./MDB'):
        # Iterate through all names in the XML
        for name in mdb.findall('.//NAMEN/NAME'):
            vorname = name.find('.//VORNAME').text or ''
            nachname = name.find('.//NACHNAME').text or ''
            adel = name.find('.//ADEL').text or ''
            praefix = name.find('.//PRAEFIX').text or ''

            # Include optional prefixes or titles
            full_first_name = f"{vorname}".strip()
            full_last_name = f"{praefix} {nachname}".strip()


            # Add any "adel" titles (e.g., "von", "de")
            if adel:
                full_last_name = f"{adel} {nachname}".strip()

            # Match using full name first
            if full_first_name == first_name and full_last_name == last_name:
                return mdb.find('.//GESCHLECHT').text

            if full_first_name == first_name:
                return mdb.find('.//GESCHLECHT').text

            # Safety net: Match only by first name if full match fails
            if primary_first_name == full_first_name:
                return mdb.find('.//GESCHLECHT').text

            if secondary_first_name == full_first_name:
                return mdb.find('.//GESCHLECHT').text

    return None


# Load the XML file
data_directory = 'data'
all_speech_data = []
nonverbal_speechacts = ["Lachen", "Widerspruch", "Beifall", "Heiterkeit", "Zurufe", "Zuruf", "Unruhe", "Zustimmung"]
verbal_speechacts = ["Zurufe", "Zuruf"]
parties = ["LINKE", "LINKEN", "DIE LINKE", "CDU/CSU", "BÜNDNIS 90/DIE GRÜNEN", "BÜNDNISSES 90/DIE GRÜNEN",
           'BÜNDNIS 90/Die Grünen', 'BÜNDIS 90/DIE GRÜNEN', "SPD", "FDP", "AfD", "Die PARTEI", "Der PARTEI", "parteilos", "LKR"]



speechact_pattern = rf"(?P<speechact>{'|'.join(nonverbal_speechacts)})"
speechact_pattern_verbal = rf"(?P<speechact>{'|'.join(verbal_speechacts)})"


party_pattern = rf"\b(?:{'|'.join(parties)})\b"
party_pattern_bracket = rf"(?:{'|'.join(parties)})"

mp_party_pattern = rf"Abg\.\s+([A-Za-zÄÖÜäöüß\-\.]+(?:\s[A-Za-zÄÖÜäöüß\-\.]+)*)\s+(?:\[[^\]]+\]\s*)?\[\s*({party_pattern_bracket})\s*\](?:\s+und\s+([A-Za-zÄÖÜäöüß\-\.]+(?:\s[A-Za-zÄÖÜäöüß\-\.]+)*)\s+(?:\[[^\]]+\]\s*)?\[\s*({party_pattern_bracket})\s*\])*"
pattern_verbal = r".*: "  # matches anything between square brackets# matches multiple speech acts in one interjection

# Matches the party of the verbal interjection
pattern_party_verbal = r"(?:\[[^\]]*\]\s*)?\[([A-Za-zÜÄÖ0-9/\s]+)\]"

# Matches the person of the verbal interjection (occurs before party)
pattern_person_verbal = r"^(.*?)\[" # match person before []
pattern_text_verbal =  r"\](.*?):\s*(.*)" # match text after ]: (or there can be something between ] and :
pattern_text_verbal_zuruf =  r"\:\s*(.*)" # match text of interjection

pattern_speechact = "|".join(nonverbal_speechacts) # all nonverbal speechacts from list

# pattern for a nonverbal interjection,  e.g. "Beifall bei der SPD", "Lachen bei Abgeordneten der AfD"
pattern_nonverbal = rf"(?P<speechact>{pattern_speechact})\s*(?:bei(?:m)?(?: den| der| des)?(?: Abgeordneten der)?\s*(?P<parties>(?:{'|'.join([f'\\b{party}\\b' for party in parties])})(?:\s*(?:,|und|sowie)\s*(?:{'|'.join([f'\\b{party}\\b' for party in parties])}))*)?)?"


## Get meta information for verbal (Zurufe)
pattern_speechact_verbal = "|".join(verbal_speechacts)
pattern_zurufe = rf"(?P<speechact>{pattern_speechact_verbal})\s+(?:von(?: den| der| des)?|vom)\s+(?P<party>{'|'.join([f'\\b{party}\\b' for party in parties])})\s*:\s*(?P<text_after_colon>.+)"
pattern_gegenruf = rf"Gegenruf\s+(?:des\s+Abg\.\s+|der\s+Abg\.\s+)?(?P<speaker>[A-Za-zÄÖÜäöüß\-\.]+(?:\s[A-Za-zÄÖÜäöüß\-\.]+)*)\s*(?:\[[^\]]*\])?\s+\[(?P<party>{party_pattern})\]\s*:\s*(?P<text>.+)"
pattern_gegenruf_nospeaker = rf"Gegenruf\s+von\s+(der|den|dem)\s+(?P<party>{party_pattern})\s*:\s*(?P<text>.+)"


# Split by '--' to capture several speech acts separately, e.g. "Beifall bei Abgeordneten der SPD – Christian Dürr [FDP]: Und Sie gar nicht? Sie waren nie dabei, Herr Daldrup?"
split_pattern = r" (?<= )[-–](?= ) "

paragraph_list = []

# Keep track of speech ID's
speech_id = 0

for filename in os.listdir(data_directory):
    file_path = os.path.join(data_directory, filename)
    tree = ET.parse(file_path)
    root = tree.getroot()

    # get date
    publication_stmt = root.find('.//teiHeader/fileDesc/publicationStmt')
    date_element = publication_stmt.find('date')
    date = date_element.text
    print(date)

    title_smt = root.find('.//teiHeader/fileDesc/titleStmt')
    period_element = title_smt.find('legislativePeriod')
    period = period_element.text

    divs = root.findall('.//div[@type="agenda_item"]')
    for div in divs:
        desc = div.get('desc')
        speeches = div.findall('.//sp')

        for sp in speeches:
            # Extract speaker information from 'name' and 'role' attributes
            speaker_role = sp.get('role')
            if speaker_role == 'mp': #: get only speeches mp's (not government)
                speech_id += 1
                speaker_name = sp.get('name')
                gender_sp = get_gender_from_name(speaker_name, sd_root)

                party = sp.get('party')

                party_name = party if party else 'Unknown'

                # Initialize the speech dictionary to store paragraphs with interjections
                speech_dict = []

                # Get all paragraphs and stage (interjection) elements within the speech
                elements = list(sp.iter())  # This gets all elements in the speech in order
                idx_element = 0

                for i, element in enumerate(elements):
                    if element.tag == 'p':  # If it's a paragraph
                        idx_element += 1
                        text = element.text
                        is_interjection = False
                        interjection = None
                        interjector = None
                        gender_int = None
                        interjector_party = None
                        is_verbal_interjection = False
                        is_nonverbal_interjection = False
                        interjection_type = None

                        paragraph_list.append({'Filename': filename,
                                               'Period': period,
                                               'Date' : date,
                                               'Item': desc,
                                               'Speech #': speech_id,
                                               'Paragraph #': idx_element,
                                               'Speaker': speaker_name,
                                               'Role': speaker_role,
                                               'Gender': gender_sp,
                                               'Party': party_name,
                                               'Paragraph': text,
                                               'Interjection': is_interjection,
                                               'Interjector' : interjector,
                                               'Interjector Gender': gender_int,
                                               'Interjector Party' : interjector_party,
                                               'Verbal interjection': is_verbal_interjection,
                                               'Nonverbal interjection': is_nonverbal_interjection,
                                               'Interjection type': interjection_type,
                                               })

                    if element.tag == 'stage':  # Next element is an interjection
                        is_interjection = True
                        interjection_nonverbal_meta = []
                        interjection_text = element.text
                        interjection_text = re.sub(r"[()]", "", interjection_text) # remove brackets around interjections

                        # Split the text using the pattern
                        parts = re.split(split_pattern, interjection_text)

                        for part in parts:
                            if re.findall(pattern_verbal, part):
                                is_verbal_interjection = True
                                is_nonverbal_interjection = False

                                match_verbal_zurufe = re.search(pattern_zurufe, part)
                                match_verbal_gegenrufe = re.search(pattern_gegenruf, part)
                                match_verbal_gegenrufe_nospeaker = re.search(pattern_gegenruf_nospeaker, part)

                                speechact_matches_verbal = list(re.finditer(speechact_pattern_verbal, part))

                                # Handle Zurufe by several parties
                                # Zurufe von der LINKEN: Pfui! – Ute Vogt [SPD]: Schämen Sie sich! – Niema Movassat [DIE LINKE]: Unerträglich!
                                if speechact_matches_verbal:
                                    for i, match in enumerate(speechact_matches_verbal):
                                        start = match.end()
                                        end = part.find(":", start)
                                        segment = part[start:end]
                                        parties_found = re.findall(party_pattern, segment)
                                        replacements = {
                                            'BÜNDNISSES 90/DIE GRÜNEN': 'BÜNDNIS 90/DIE GRÜNEN',
                                            'LINKEN': 'LINKE',
                                            'DIE LINKE': 'LINKE',
                                        }
                                        #rep
                                        text_zuruf = re.search(pattern_text_verbal_zuruf, part)
                                        annotation = []
                                        for party in parties_found:
                                            interjector_party = party
                                            interjector = "Unknown"
                                            text = text_zuruf.group(1)
                                            interjection_type = "Zuruf"
                                            paragraph_list.append({'Filename': filename,
                                                                   'Period': period,
                                                                   'Date' : date,
                                                                   'Item': desc,
                                                                   'Speech #': speech_id,
                                                                   'Paragraph #': idx_element,
                                                                   'Speaker': speaker_name,
                                                                   'Role': speaker_role,
                                                                   'Gender': gender_sp,
                                                                   'Party': party_name,
                                                                   'Paragraph': text,
                                                                   'Interjection': is_interjection,
                                                                   'Interjector' : interjector,
                                                                   'Interjector Party' : interjector_party,
                                                                   'Verbal interjection': is_verbal_interjection,
                                                                   'Nonverbal interjection': is_nonverbal_interjection,
                                                                   'Interjection type': interjection_type,
                                                                   })

                                # Gegenrufe (speaker known -> 'Gegenruf des Abg. Volker Kauder [CDU/CSU]: Das auch!')
                                elif match_verbal_gegenrufe:
                                    interjector = match_verbal_gegenrufe.group("speaker")
                                    if ',' in interjector:
                                        interjector = interjector.split(',')[0].strip()
                                    names = interjector.split()
                                    names = " ".join(names)
                                    if names:
                                        last_name_int = names[-1]
                                        first_name_int = " ".join(names[:-1])

                                        gender_int = get_gender_from_name(names, sd_root)

                                        # if none is found, try different combination (two last names, e.g. Amira Mohamed Ali)
                                        if not gender_int:
                                            first_name_int = names[0]
                                            last_name_int = " ".join(names[1:])
                                            gender_int = get_gender_from_name(names, sd_root)
                                    else:
                                        gender_int = None


                                    interjector_party = match_verbal_gegenrufe.group("party")
                                    text = match_verbal_gegenrufe.group("text")
                                    interjection_type = "Gegenruf"

                                    paragraph_list.append({'Filename': filename,
                                                           'Period': period,
                                                           'Date' : date,
                                                           'Item': desc,
                                                           'Speech #': speech_id,
                                                           'Paragraph #': idx_element,
                                                           'Speaker': speaker_name,
                                                           'Role': speaker_role,
                                                           'Gender': gender_sp,
                                                           'Party': party_name,
                                                           'Paragraph': text,
                                                           'Interjection': is_interjection,
                                                           'Interjector' : interjector,
                                                           'Interjector Gender': gender_int,
                                                           'Interjector Party' : interjector_party,
                                                           'Verbal interjection': is_verbal_interjection,
                                                           'Nonverbal interjection': is_nonverbal_interjection,
                                                           'Interjection type': interjection_type,
                                                           })

                                elif match_verbal_gegenrufe_nospeaker:
                                    interjector = "Unknown"
                                    gender_int = None
                                    interjector_party = match_verbal_gegenrufe_nospeaker.group("party")
                                    text = match_verbal_gegenrufe_nospeaker.group("text")
                                    interjection_type = "Gegenruf"

                                    paragraph_list.append({'Filename': filename,
                                                           'Period': period,
                                                           'Date' : date,
                                                           'Item': desc,
                                                           'Speech #': speech_id,
                                                           'Paragraph #': idx_element,
                                                           'Speaker': speaker_name,
                                                           'Role': speaker_role,
                                                           'Gender': gender_sp,
                                                           'Party': party_name,
                                                           'Paragraph': text,
                                                           'Interjection': is_interjection,
                                                           'Interjector' : interjector,
                                                           'Interjector Gender': gender_int,
                                                           'Interjector Party' : interjector_party,
                                                           'Verbal interjection': is_verbal_interjection,
                                                           'Nonverbal interjection': is_nonverbal_interjection,
                                                           'Interjection type': interjection_type,
                                                           })

                                else:
                                    # extract speaker party
                                    party_match = re.search(pattern_party_verbal, part)
                                    if party_match:
                                        interjector_party = party_match.group(1).strip()
                                    else:
                                        interjector_party = 'Unknown'

                                    # extract speaker name
                                    person_match = re.search(pattern_person_verbal, part)
                                    if person_match:
                                        interjector= person_match.group(1).strip()
                                        if ',' in interjector:
                                            interjector = interjector.split(',')[0].strip()
                                        names = interjector.split()
                                        names = " ".join(names)

                                        if names:
                                            last_name_int = names[-1]
                                            first_name_int = " ".join(names[:-1])
                                            gender_int = get_gender_from_name(names, sd_root)


                                            # if none is found, try different combination (two last names, e.g. Amira Mohamed Ali)
                                            if not gender_int:
                                                first_name_int = names[0]
                                                last_name_int = " ".join(names[1:])
                                                gender_int = get_gender_from_name(names, sd_root)
                                        else:
                                            gender_int = None

                                    else:
                                        interjector = 'Unknown'

                                    # extract text only
                                    text_match = re.search(pattern_text_verbal, part)
                                    if text_match:
                                        extra_text = text_match.group(1).strip
                                        text = text_match.group(2).strip()
                                    else:
                                        text = None

                                    interjection_type = "Zuruf"


                                    paragraph_list.append({'Filename': filename,
                                                           'Period': period,
                                                           'Date' : date,
                                                           'Item': desc,
                                                           'Speech #': speech_id,
                                                           'Paragraph #': idx_element,
                                                           'Speaker': speaker_name,
                                                           'Role': speaker_role,
                                                           'Gender': gender_sp,
                                                           'Party': party_name,
                                                           'Paragraph': text,
                                                           'Interjection': is_interjection,
                                                           'Interjector' : interjector,
                                                           'Interjector Gender': gender_int,
                                                           'Interjector Party' : interjector_party,
                                                           'Verbal interjection': is_verbal_interjection,
                                                           'Nonverbal interjection': is_nonverbal_interjection,
                                                           'Interjection type': interjection_type,
                                                           })


                            else:
                                is_verbal_interjection = False
                                is_nonverbal_interjection = True
                                speechact_matches = list(re.finditer(speechact_pattern, part))
                                if speechact_matches:
                                    results = []
                                    for i, match in enumerate(speechact_matches):
                                        start = match.end()
                                        end = speechact_matches[i + 1].start() if i + 1 < len(speechact_matches) else len(part)
                                        segment = part[start:end]
                                        parties_found = re.findall(party_pattern, segment)

                                        matches_mp = re.findall(mp_party_pattern, segment)

                                        mps_found = []

                                        if matches_mp:
                                            first_name, first_party, *rest = matches_mp[0]
                                            mps_found.append((first_name, first_party))

                                            for i in range(0, len(rest), 2):
                                                name = rest[i]
                                                party = rest[i + 1]
                                                if name and party:
                                                    mps_found.append((name, party))
                                        replacements = {
                                            'BÜNDNISSES 90/DIE GRÜNEN': 'BÜNDNIS 90/DIE GRÜNEN',
                                            'LINKEN': 'LINKE'
                                        }
                                        #rep
                                        for party in parties_found:
                                            for old, new in replacements.items():
                                                party = re.sub(rf'\b{old}\b', new, party)

                                            individual_added = False
                                            few_mps = "Abgeordneten" in segment # check if ALL abgeordnete

                                            if mps_found:
                                                for interjector, interjector_party in mps_found:
                                                    names = interjector.split()
                                                    names = " ".join(names)
                                                    last_name_int = names[-1]
                                                    first_name_int = " ".join(names[:-1])
                                                    gender_int = get_gender_from_name(names,
                                                                                      sd_root)
                                                    if interjector_party == party:
                                                        interjection_type = match.group("speechact")
                                                        if (interjector_party, interjector, interjection_type) not in interjection_nonverbal_meta:
                                                            if interjection_type in ["Zuruf", "Zurufe", "Widerspruch", "Zustimmung"]:
                                                                paragraph_list.append({'Filename': filename,
                                                                                       'Period': period,
                                                                                       'Date' : date,
                                                                                       'Item': desc,
                                                                                       'Speech #': speech_id,
                                                                                       'Paragraph #': idx_element,
                                                                                       'Speaker': speaker_name,
                                                                                       'Role': speaker_role,
                                                                                       'Gender': gender_sp,
                                                                                       'Party': party_name,
                                                                                       'Paragraph': None,
                                                                                       'Interjection': is_interjection,
                                                                                       'Interjector' : interjector,
                                                                                       'Interjector Gender': gender_int,
                                                                                       'Interjector Party' : interjector_party,
                                                                                       'Verbal interjection': True,
                                                                                       'Nonverbal interjection': False,
                                                                                       'Interjection type': interjection_type,
                                                                                       })
                                                            else:
                                                                paragraph_list.append({'Filename': filename,
                                                                                       'Period': period,
                                                                                       'Date' : date,
                                                                                       'Item': desc,
                                                                                       'Speech #': speech_id,
                                                                                       'Paragraph #': idx_element,
                                                                                       'Speaker': speaker_name,
                                                                                       'Role': speaker_role,
                                                                                       'Gender': gender_sp,
                                                                                       'Party': party_name,
                                                                                       'Paragraph': None,
                                                                                       'Interjection': is_interjection,
                                                                                       'Interjector' : interjector,
                                                                                       'Interjector Gender': gender_int,
                                                                                       'Interjector Party' : interjector_party,
                                                                                       'Verbal interjection': False,
                                                                                       'Nonverbal interjection': True,
                                                                                       'Interjection type': interjection_type,
                                                                                       })

                                                        individual_added = True
                                            if not individual_added:
                                                interjection_type = match.group("speechact")
                                                if interjection_type in ["Zuruf", "Zurufe", "Widerspruch", "Zustimmung"]:
                                                    paragraph_list.append({'Filename': filename,
                                                                           'Period': period,
                                                                           'Date' : date,
                                                                           'Item': desc,
                                                                           'Speech #': speech_id,
                                                                           'Paragraph #': idx_element,
                                                                           'Speaker': speaker_name,
                                                                           'Role': speaker_role,
                                                                           'Gender': gender_sp,
                                                                           'Party': party_name,
                                                                           'Paragraph': None,
                                                                           'Interjection': is_interjection,
                                                                           'Interjector' : 'Unknown',
                                                                           'Interjector Gender': gender_int,
                                                                           'Interjector Party' : party,
                                                                           'Verbal interjection': True,
                                                                           'Nonverbal interjection': False,
                                                                           'Interjection type': interjection_type,
                                                                           })
                                                else:
                                                    paragraph_list.append({'Filename': filename,
                                                                           'Period': period,
                                                                           'Date' : date,
                                                                           'Item': desc,
                                                                           'Speech #': speech_id,
                                                                           'Paragraph #': idx_element,
                                                                           'Speaker': speaker_name,
                                                                           'Role': speaker_role,
                                                                           'Gender': gender_sp,
                                                                           'Party': party_name,
                                                                           'Paragraph': None,
                                                                           'Interjection': is_interjection,
                                                                           'Interjector' : 'all',
                                                                           'Interjector Gender': gender_int,
                                                                           'Interjector Party' : party,
                                                                           'Verbal interjection': False,
                                                                           'Nonverbal interjection': True,
                                                                           'Interjection type': interjection_type,
                                                                           })

                                        if not parties_found:
                                            interjection_type = match.group("speechact")
                                            if interjection_type in ["Zuruf", "Zurufe", "Widerspruch", "Zustimmung"]:
                                                paragraph_list.append({'Filename': filename,
                                                                       'Period': period,
                                                                       'Date' : date,
                                                                       'Item': desc,
                                                                       'Speech #': speech_id,
                                                                       'Paragraph #': idx_element,
                                                                       'Speaker': speaker_name,
                                                                       'Role': speaker_role,
                                                                       'Gender': gender_sp,
                                                                       'Party': party_name,
                                                                       'Paragraph': None,
                                                                       'Interjection': is_interjection,
                                                                       'Interjector' : 'all',
                                                                       'Interjector Gender': gender_int,
                                                                       'Interjector Party' : 'all',
                                                                       'Verbal interjection': True,
                                                                       'Nonverbal interjection': False,
                                                                       'Interjection type': interjection_type,
                                                                       })
                                            else:
                                                paragraph_list.append({'Filename': filename,
                                                                       'Period': period,
                                                                       'Date' : date,
                                                                       'Item': desc,
                                                                       'Speech #': speech_id,
                                                                       'Paragraph #': idx_element,
                                                                       'Speaker': speaker_name,
                                                                       'Role': speaker_role,
                                                                       'Gender': gender_sp,
                                                                       'Party': party_name,
                                                                       'Paragraph': None,
                                                                       'Interjection': is_interjection,
                                                                       'Interjector' : 'all',
                                                                       'Interjector Gender': gender_int,
                                                                       'Interjector Party' : 'all',
                                                                       'Verbal interjection': False,
                                                                       'Nonverbal interjection': True,
                                                                       'Interjection type': interjection_type,
                                                                       })

speeches_df = pd.DataFrame(paragraph_list)


# Exception: Lachen des Abg. Dr. h. c. [Univ Kyiv] Hans Michelbach [CDU/CSU], need to manually add because [Univ Kyiv]
# is interpreted as party
# Standardize party names

speeches_df['Party'] = speeches_df['Party'].replace({'CDU': 'CDU/CSU', 'CSU': 'CDU/CSU', 'Univ Kyiv': 'CDU/CSU',
                                   'UnivKyiv':'CDU/CSU', 'Erlangen':'CDU/CSU', 'BÜNDNIS 90/D': 'GRUENE',
                                   'BÜNDNISSES 90/DIE GRÜNEN': 'GRUENE','BÜNDNIS 90/DIE GRÜNEN': 'GRUENE',
                                   'BÜNDIS 90/DIE GRÜNEN': 'GRUENE', 'LINKEN': 'DIE LINKE', 'LINKE': 'DIE LINKE'})

speeches_df['Interjector Party'] = speeches_df['Interjector Party'].replace({'CDU': 'CDU/CSU', 'CSU': 'CDU/CSU', 'Univ Kyiv': 'CDU/CSU',
                                                           'UnivKyiv':'CDU/CSU', 'Erlangen':'CDU/CSU',
                                                           'BÜNDNIS 90/D': 'GRUENE', 'BÜNDNISSES 90/DIE GRÜNEN': 'GRUENE',
                                                           'BÜNDNIS 90/DIE GRÜNEN': 'GRUENE', 'BÜNDIS 90/DIE GRÜNEN': 'GRUENE',
                                                           'LINKEN': 'DIE LINKE', 'LINKE': 'DIE LINKE'})
speeches_df.to_csv('new_speeches_output.csv', index=False )
print("Data successfully saved to 'new_speeches_output.csv'")
