import os
from tqdm import tqdm
import re
import regex
import xml.etree.ElementTree as ET
import pandas as pd
import itertools

# refer to stammdaten to get gender
stammdaten = "MdB-Stammdaten/MDB_STAMMDATEN.XML"
sd_tree = ET.parse(stammdaten)
sd_root = sd_tree.getroot()


# build index once
def build_mdb_index(root):
    index = {}
    fallback_by_first = {}
    fallback_by_last = {}

    for mdb in tqdm(root.findall('./MDB'), desc="Building index", unit="MDB"):
        for name in mdb.findall('.//NAMEN/NAME'):
            vorname = name.find('.//VORNAME').text or ''
            nachname = name.find('.//NACHNAME').text or ''
            adel = name.find('.//ADEL').text or ''
            praefix = name.find('.//PRAEFIX').text or ''

            full_first_name = " ".join(vorname.strip().split())
            full_last_name = " ".join(f"{adel} {praefix} {nachname}".strip().split())
            full_name = f"{full_first_name} {full_last_name}".lower().strip()

            gender = mdb.find('.//GESCHLECHT').text
            party = mdb.find('.//PARTEI_KURZ').text
            periods = {wp.find('.//WP').text for wp in mdb.findall('.//WAHLPERIODEN/WAHLPERIODE')}

            entry = {
                "gender": gender,
                "party": party,
                "name": f"{full_first_name} {full_last_name}",
                "periods": periods
            }

            index[full_name] = entry

            fallback_by_first.setdefault(full_first_name.lower(), []).append(entry)
            fallback_by_last.setdefault(full_last_name.lower(), []).append(entry)

    return index, fallback_by_first, fallback_by_last

index, fallback_by_first, fallback_by_last = build_mdb_index(sd_root)
print(index)

def get_gender_from_name(full_name, index, fallback_by_first, fallback_by_last, period):
    def normalize_name(name):
        titles_to_remove = ["Dr.", "Dr. h. c.", "Prof.", "h. c."]
        for title in titles_to_remove:
            name = name.replace(title, "")
        return name.strip()

    normalized = normalize_name(full_name).lower()
    print(normalized)
    names = normalized.split()
    first = names[0] if names else ""
    second = names[1] if len(names) > 1 else ""
    last = names[-1] if names else ""

    # 1. Exact match, check if name + legislature period match
    if normalized in index:
        entry = index[normalized]
        if str(period) in entry["periods"]:
            return entry

    # 2. Fallback: by last name, check if last name + legislature period match
    for entry in fallback_by_last.get(last.lower(), []):
        if str(period) in entry["periods"]:
            return entry

    # 3. Fallback: Match by first or second first name, check if name + legislature period match
    for fname in (first, second):
        for entry in fallback_by_first.get(fname.lower(), []):
            if str(period) in entry["periods"]:
                return entry

    return None



# Load the XML file
data_directory = 'data_test'
all_speech_data = []
no_colon_speechacts = ["Lachen", "Widerspruch", "Beifall", "Heiterkeit", "Zurufe", "Zuruf", "Unruhe", "Zustimmung"]
colon_speechacts = ["Zurufe", "Zuruf"]
parties = ["LINKE", "LINKEN", "DIE LINKE", "Linken", "CDU/CSU", "CDU/ CSU", "BÜNDNIS 90/DIE GRÜNEN", "BÜNDNISSES 90/DIE GRÜNEN",
           'BÜNDNIS 90/Die Grünen', 'GRÜNE', 'BÜNDIS 90/DIE GRÜNEN', "SPD", "FDP", "FPD", "AfD", "Die PARTEI", "Der PARTEI", "parteilos", "LKR",
           "DP", "CVP", "GB/ BHE", "GB/BHE", "DA", "DZP"]



speechact_pattern_no_colon = (
    rf"(?P<speechact>(?:{'|'.join(no_colon_speechacts)})(?:\s+(?:und)\s+(?:{'|'.join(no_colon_speechacts)}))*)(?!\s*:)"
)
speechact_pattern_colon = rf"(?P<speechact>{'|'.join(colon_speechacts)})"


party_pattern = rf"\b(?:{'|'.join(parties)})\b"
party_pattern_bracket = rf"(?:{'|'.join(parties)})"


mp_party_pattern = r"(?:Abg\.|Bundesminister(?:in)?s?)\s+([\p{L}\w\-\.]+(?:\s[\p{L}\w\-\.]+)*)\s+(?:\[[^\]]+\]\s*)?\[\s*(" + party_pattern_bracket + r")\s*\](?:\s+und\s+([\p{L}\w\-\.]+(?:\s[\p{L}\w\-\.]+)*)\s+(?:\[[^\]]+\]\s*)?\[\s*(" + party_pattern_bracket + r")\s*\])*"


pattern_colon = r".*: "  # matches anything between square brackets# matches multiple speech acts in one interjection
# Matches the party of the interjection with a colon
pattern_party_colon = r"(?:\[[^\]]*\]\s*)?\[([A-Za-zÜÄÖ0-9/\s]*[A-Z])\]"
# Matches the person of the interjection with a colon(occurs before party)
pattern_person_colon = r"^(?:Abg\.\s*)?(.*?)(?:,\s*Bundesminister(?:in)?)?\s*(?:\[|:)"
pattern_text_colon =  r"\]?(.*?):\s*(.*)" # match text after potentially ]: (or there can be something between ] and : or just :
pattern_text_colon_zuruf =  r"\:\s*(.*)" # match text of interjection
pattern_speechact = "|".join(no_colon_speechacts) # all speechacts from list NOT preceding a colon


## Get meta information for colon speechacts (Zurufe)
pattern_speechact_colon = "|".join(colon_speechacts)

# catch multiple parties in "Zurufe"
multi_party_pattern = rf"(?:den|der|des|dem)?\s*{party_pattern_bracket}(?:\s+und\s+(?:den|der|des|dem)?\s*{party_pattern_bracket})*"
pattern_zurufe = rf"(?P<speechact>{pattern_speechact_colon})\s+(?:(von|vom)\s+(?:Abgeordneten\s+)?)?(?P<party>{multi_party_pattern})\s*:\s*(?P<text_after_colon>.+)"

pattern_gegenruf = rf"(?P<interjection_type>Gegenruf[e]?)\s+(?:des\s+Abg\.\s+|der\s+Abg\.\s+)?(?P<speaker>[A-Za-zÄÖÜäöüß\-\.]+(?:\s[A-Za-zÄÖÜäöüß\-\.]+)*)\s*(?:\[[^\]]*\])?\s+\[(?P<party>{party_pattern})\]\s*:\s*(?P<text>.+)"
pattern_gegenruf_nospeaker = rf"(?P<interjection_type>Gegenruf[e]?)\s+(?:von (?:dem|der|den)|vom)\s+(?:(?P<party>{party_pattern})|(?P<speaker>[^\:]+))\s*:\s*(?P<text>.+)"

interjection_patterns = [
    ("Gegenruf_with_speaker", pattern_gegenruf),
    ("Gegenruf_without_speaker", pattern_gegenruf_nospeaker),
    ("Zuruf_colon", pattern_zurufe),
]

# Split by '--—' to capture several speech acts separately, e.g. "Beifall bei Abgeordneten der SPD – Christian Dürr [FDP]: Und Sie gar nicht? Sie waren nie dabei, Herr Daldrup?"
split_pattern = r" (?<= )[-–—](?= ) "

paragraph_list = []

# Keep track of speech ID's
speech_id = 0

for filename in tqdm(os.listdir(data_directory), desc="Processing files", unit="file"):
    file_path = os.path.join(data_directory, filename)
    tree = ET.parse(file_path)
    root = tree.getroot()

    # get date
    publication_stmt = root.find('.//teiHeader/fileDesc/publicationStmt')
    date_element = publication_stmt.find('date')
    date = date_element.text
    print(filename)

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
                gender_sp = get_gender_from_name(speaker_name, index, fallback_by_first, fallback_by_last, period)["gender"]
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

                        # Split the text at -–—
                        parts = re.split(split_pattern, interjection_text)

                        for part in parts:
                            print(part)

                            ### ONLY VERBAL INTERJECTIONS ("Widerspruch", "Gegenruf", "Gegenrufe", "Zuruf", "Zustimmung") WITH COLON
                            if re.findall(pattern_colon, part):
                                is_verbal_interjection = True
                                is_nonverbal_interjection = False

                                matched_any = False

                                for label, rege in interjection_patterns:
                                    for match in re.finditer(rege, part):
                                        matched_any = True
                                        interjection_type = match.group(
                                            "interjection_type") if "interjection_type" in match.groupdict() else match.group(
                                            "speechact")
                                        if "speaker" in match.groupdict() and match.group("speaker"):
                                            interjector = match.group("speaker")
                                        elif "abgeordneten" in match.group(0).lower():
                                            interjector = "some"
                                        else:
                                            interjector = "Unknown"

                                        # Process each matched party (including when "und" appears)
                                        parties_found = re.findall(party_pattern, part)
                                        if parties_found:
                                            for interjector_party in parties_found:
                                                replacements = {
                                                    'BÜNDNISSES 90/DIE GRÜNEN': 'BÜNDNIS 90/DIE GRÜNEN',
                                                    'LINKEN': 'LINKE',
                                                    'Linken': 'LINKE',
                                                    'FPD':'FDP'

                                                }

                                                for old, new in replacements.items():
                                                    interjector_party = re.sub(rf'\b{old}\b', new, interjector_party)

                                                # If "party" is found, update the interjector party
                                                interjector_party = interjector_party.strip()

                                                text = match.group(
                                                    "text_after_colon") if "text_after_colon" in match.groupdict() else match.group(
                                                    "text")
                                                gender_int = None
                                                if interjector not in ["Unknown", "some"]:
                                                    names = " ".join(interjector.split())
                                                    meta_info = get_gender_from_name(names, index, fallback_by_first,
                                                                                     fallback_by_last, period)

                                                    print("CHECK", names)
                                                    print(meta_info)

                                                    if meta_info:
                                                        interjector = meta_info["name"]
                                                        gender_int = meta_info["gender"]


                                                        if interjector_party == "Unknown" or interjector_party[
                                                            -1].islower():
                                                            interjector_party = meta_info["party"]


                                                paragraph_list.append({
                                                    'Filename': filename,
                                                    'Period': period,
                                                    'Date': date,
                                                    'Item': desc,
                                                    'Speech #': speech_id,
                                                    'Paragraph #': idx_element,
                                                    'Speaker': speaker_name,
                                                    'Role': speaker_role,
                                                    'Gender': gender_sp,
                                                    'Party': party_name,
                                                    'Paragraph': text,
                                                    'Interjection': is_interjection,
                                                    'Interjector': interjector,
                                                    'Interjector Gender': gender_int,
                                                    'Interjector Party': interjector_party,
                                                    'Verbal interjection': is_verbal_interjection,
                                                    'Nonverbal interjection': is_nonverbal_interjection,
                                                    'Interjection type': interjection_type,
                                                })

                                        else:
                                            interjector_party = "Unknown"
                                            gender_int = None
                                            text = match.group(
                                                "text_after_colon") if "text_after_colon" in match.groupdict() else match.group(
                                                "text")

                                            paragraph_list.append({
                                                'Filename': filename,
                                                'Period': period,
                                                'Date': date,
                                                'Item': desc,
                                                'Speech #': speech_id,
                                                'Paragraph #': idx_element,
                                                'Speaker': speaker_name,
                                                'Role': speaker_role,
                                                'Gender': gender_sp,
                                                'Party': party_name,
                                                'Paragraph': text,
                                                'Interjection': is_interjection,
                                                'Interjector': interjector,
                                                'Interjector Gender': gender_int,
                                                'Interjector Party': interjector_party,
                                                'Verbal interjection': is_verbal_interjection,
                                                'Nonverbal interjection': is_nonverbal_interjection,
                                                'Interjection type': interjection_type,
                                            })

                                # Fallback: basic speaker:party:text if no match above
                                if not matched_any:
                                    person_match = re.search(pattern_person_colon, part)
                                    party_match = re.search(pattern_party_colon, part)
                                    text_match = re.search(pattern_text_colon, part)

                                    interjector = person_match.group(1).strip() if person_match else "Unknown"
                                    interjector_party = party_match.group(1).strip() if party_match else "Unknown"
                                    text = text_match.group(2).strip() if text_match else None
                                    interjection_type = "Zuruf"

                                    gender_int = None
                                    if interjector != "Unknown":
                                        names = " ".join(interjector.split())
                                        meta_info = get_gender_from_name(names, index, fallback_by_first,
                                                                         fallback_by_last, period)

                                        print("CHECK", names)
                                        print(meta_info)

                                        if meta_info:
                                            interjector = meta_info["name"]
                                            gender_int = meta_info["gender"]
                                            if interjector_party == "Unknown" or interjector_party[-1].islower():
                                                interjector_party = meta_info["party"]


                                    paragraph_list.append({
                                        'Filename': filename,
                                        'Period': period,
                                        'Date': date,
                                        'Item': desc,
                                        'Speech #': speech_id,
                                        'Paragraph #': idx_element,
                                        'Speaker': speaker_name,
                                        'Role': speaker_role,
                                        'Gender': gender_sp,
                                        'Party': party_name,
                                        'Paragraph': text,
                                        'Interjection': is_interjection,
                                        'Interjector': interjector,
                                        'Interjector Gender': gender_int,
                                        'Interjector Party': interjector_party,
                                        'Verbal interjection': is_verbal_interjection,
                                        'Nonverbal interjection': is_nonverbal_interjection,
                                        'Interjection type': interjection_type,
                                    })

                            ### CAN BE NONVERBAL OR VERBAL INTERJECTIONS "Lachen", "Widerspruch", "Beifall",
                            # "Heiterkeit", "Zurufe", "Zuruf", "Unruhe", "Zustimmung", "Ruf", "Rufe", "Gegenruf", "Gegenrufe"
                            #  WITHOUT COLON
                            else:
                                is_verbal_interjection = False
                                is_nonverbal_interjection = True
                                speechact_matches = list(re.finditer(speechact_pattern_no_colon, part))

                                print(f"Speechact_matches: {speechact_matches}")

                                if speechact_matches:
                                    results = []
                                    for i, match in enumerate(speechact_matches):
                                        print(speechact_matches)

                                        start = match.end()

                                        end = speechact_matches[i + 1].start() if i + 1 < len(
                                            speechact_matches) else len(part)

                                        segment = part[start:end]

                                        split_sowie = r'\bsowie\b'
                                        print(f"segment:", {segment})
                                        parts_sowie = re.split(split_sowie, segment)
                                        print(f"Parts_sowie: {parts_sowie}")
                                        for part_sowie in parts_sowie:
                                            few_mps = "Abgeordneten" in part_sowie
                                            parties_found = re.findall(party_pattern, part_sowie)
                                            matches_mp = regex.findall(mp_party_pattern, part_sowie)
                                            mps_found = []
                                            print("MATCHES MP", matches_mp)

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
                                                'LINKEN': 'LINKE',
                                                'Linken': 'LINKE',
                                                 'FPD':'FDP'

                                            }
                                            print(f"mpps found: {mps_found}")

                                            for party in parties_found:
                                                few_mps = "Abgeordneten" in part_sowie  # check if ALL abgeordnete
                                                for old, new in replacements.items():
                                                    party = re.sub(rf'\b{old}\b', new, party)
                                                individual_added = False
                                                if mps_found:
                                                    print(f"mps: {mps_found}")
                                                    for interjector, interjector_party in mps_found:
                                                        names = interjector.split()
                                                        names = " ".join(names)
                                                        last_name_int = names[-1]
                                                        first_name_int = " ".join(names[:-1])
                                                        meta_info = get_gender_from_name(names, index, fallback_by_first,
                                                                                         fallback_by_last, period)
                                                        gender_int = meta_info["gender"]
                                                        print("CHECK", names)
                                                        print(meta_info)

                                                        if interjector_party == party:
                                                            interjection_type = match.group("speechact")

                                                            if (interjector_party, interjector,
                                                                interjection_type) not in interjection_nonverbal_meta:

                                                                interjector = meta_info["name"]
                                                                is_verbal = interjection_type in ["Zuruf", "Zurufe", "Widerspruch",
                                                                                         "Zustimmung", "Ruf", "Rufe"]

                                                                paragraph_list.append({'Filename': filename,
                                                                                                   'Period': period,
                                                                                                   'Date': date,
                                                                                                   'Item': desc,
                                                                                                   'Speech #': speech_id,
                                                                                                   'Paragraph #': idx_element,
                                                                                                   'Speaker': speaker_name,
                                                                                                   'Role': speaker_role,
                                                                                                   'Gender': gender_sp,
                                                                                                   'Party': party_name,
                                                                                                   'Paragraph': None,
                                                                                                   'Interjection': is_interjection,
                                                                                                   'Interjector': interjector,
                                                                                                   'Interjector Gender': gender_int,
                                                                                                   'Interjector Party': interjector_party,
                                                                                                   'Verbal interjection': True if is_verbal else False,
                                                                                                   'Nonverbal interjection': False if is_verbal else True,
                                                                                                   'Interjection type': interjection_type,
                                                                                                   })
                                                                individual_added = True

                                                if not individual_added:
                                                    print("HELP")
                                                    interjection_types = re.split(r'\s+(?:und)\s+', match.group("speechact"))
                                                    for interjection_type in interjection_types:
                                                        is_verbal = interjection_type in ["Zuruf", "Zurufe", "Widerspruch",
                                                                                      "Zustimmung", "Ruf", "Rufe"]
                                                        paragraph_list.append({'Filename': filename,
                                                                                       'Period': period,
                                                                                       'Date': date,
                                                                                       'Item': desc,
                                                                                       'Speech #': speech_id,
                                                                                       'Paragraph #': idx_element,
                                                                                       'Speaker': speaker_name,
                                                                                       'Role': speaker_role,
                                                                                       'Gender': gender_sp,
                                                                                       'Party': party_name,
                                                                                       'Paragraph': None,
                                                                                       'Interjection': is_interjection,
                                                                                       'Interjector': (
                                                                                            'some' if few_mps and (
                                                                                                (not is_verbal) or (interjection_type or "").lower() in ['zurufe', 'gegenrufe', 'rufe', 'widerspruch', 'zustimmung']
                                                                                            ) else (
                                                                                                'Unknown' if (interjection_type or "").lower() in ['zuruf', 'gegenruf', 'ruf']
                                                                                                else 'all'
                                                                                            )
                                                                                        ),
                                                                                       'Interjector Gender': None,
                                                                                       'Interjector Party': party,
                                                                                       'Verbal interjection': True if is_verbal else False,
                                                                                       'Nonverbal interjection': False if is_verbal else True,
                                                                                       'Interjection type': interjection_type,
                                                                                       })
                                            if not parties_found:
                                                last_end = speechact_matches[-1].end() if speechact_matches else 0
                                                trailing_text = part[last_end:]

                                                if "im ganzen Hause" in trailing_text:
                                                    parties_found = ['all']
                                                else:
                                                    found_parties = re.findall(party_pattern, trailing_text)
                                                    if found_parties:
                                                        parties_found = found_parties
                                            if not parties_found:
                                                interjection_type = match.group("speechact")
                                                is_verbal = interjection_type in ["Zuruf", "Zurufe", "Widerspruch",
                                                                                  "Zustimmung", "Ruf", "Rufe"]
                                                paragraph_list.append({'Filename': filename,
                                                                       'Period': period,
                                                                       'Date': date,
                                                                       'Item': desc,
                                                                       'Speech #': speech_id,
                                                                       'Paragraph #': idx_element,
                                                                       'Speaker': speaker_name,
                                                                       'Role': speaker_role,
                                                                       'Gender': gender_sp,
                                                                       'Party': party_name,
                                                                       'Paragraph': None,
                                                                       'Interjection': is_interjection,
                                                                       'Interjector': 'Unknown' if is_verbal else 'all',
                                                                       'Interjector Gender': None,
                                                                       'Interjector Party': 'Unknown' if is_verbal else 'all',
                                                                       'Verbal interjection': True if is_verbal else False,
                                                                       'Nonverbal interjection': False if is_verbal else True,
                                                                       'Interjection type': interjection_type,
                                                                       })


speeches_df = pd.DataFrame(paragraph_list)


speeches_df['Party'] = speeches_df['Party'].replace({'CDU': 'CDU/CSU', 'CSU': 'CDU/CSU', 'Univ Kyiv': 'CDU/CSU',
                                   'UnivKyiv':'CDU/CSU', 'Erlangen':'CDU/CSU', 'BÜNDNIS 90/D': 'GRUENE',
                                   'BÜNDNISSES 90/DIE GRÜNEN': 'GRUENE','BÜNDNIS 90/DIE GRÜNEN': 'GRUENE',
                                   'BÜNDIS 90/DIE GRÜNEN': 'GRUENE', 'LINKEN': 'DIE LINKE', 'LINKE': 'DIE LINKE'})

speeches_df['Interjector Party'] = speeches_df['Interjector Party'].replace({'CDU': 'CDU/CSU', 'CSU': 'CDU/CSU', 'CDU/ CSU': 'CDU/CSU',
                                                                             'Univ Kyiv': 'CDU/CSU','UnivKyiv':'CDU/CSU', 'Erlangen':'CDU/CSU',
                                                           'BÜNDNIS 90/D': 'GRUENE', 'BÜNDNISSES 90/DIE GRÜNEN': 'GRUENE',
                                                           'BÜNDNIS 90/DIE GRÜNEN': 'GRUENE', 'BÜNDIS 90/DIE GRÜNEN': 'GRUENE',
                                                            'GRÜNE': 'GRUENE', 'GRÜNEN': 'GRUENE', 'LINKEN': 'DIE LINKE', 'LINKE': 'DIE LINKE',
                                                            'GB/ BHE': 'GB/BHE'})
speeches_df.to_csv('test_output.csv', index=False )

print(index.get("alexander lambsdorff"))
print(index.get("alexander graf lambsdorff"))

print("Data successfully saved to 'new_speeches_output.csv'")