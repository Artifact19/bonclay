"""Hinglish: Hindi typed in Roman letters ("kal subah aana", "bhai price kitna hai?").

Hindi voices read Devanagari, so Roman Hindi has to be converted before it is spoken. Mechanical ITRANS rules turn
casual spelling into nonsense ("kal subah aana" -> "कल् सुबह् आन"), so the conversion is layered:

1. `to_devanagari` asks the language model (Groq, a local server or the MLX model) to rewrite the text, with a strict
   prompt and a few examples, then validates the answer (mostly Devanagari, similar word count, no chatbot reply).
2. Known words are then pinned to a hand-written dictionary (theek -> ठीक, appointment -> अपॉइंटमेंट), which also
   repairs any Latin leftovers the model let through.
3. With no model, or an answer that fails validation, the dictionary does the whole job, and a small Hinglish-aware
   rule transliterator handles words that are not in it.
"""
import logging
import re
import threading
from collections import OrderedDict
from typing import Dict, List, Optional

log = logging.getLogger("bonclay.hinglish")

try:
    from services import llm
except ImportError:  # pragma: no cover
    from backend.services import llm

# ---------------------------------------------------------------------------------------------- detection

# Common Hindi function words and verbs in Roman spelling. Words that are also everyday English (me, to, main, do,
# hi, the, par, pe) are left out on purpose, so plain English never looks like Hinglish.
HINGLISH_MARKERS = frozenset("""
hai hain hoon hun ho hoga hogi honge tha thi thin raha rahi rahe gaya gayi gaye kiya kiye diya liya hua hui hue
kya kyun kyon kyunki kaise kaisa kaisi kab kahan kaun kitna kitni kitne kaunsa
nahi nahin nhi haan accha acha achha theek thik sahi bilkul zaroor
mujhe mujhko mera meri mere hum humein hamara hamari tum tumhara tumhari aap aapko aapka aapki aapke aapse
yeh ye woh wo yahan wahan
kal aaj abhi jaldi subah shaam raat baad pehle phir bahut bohot thoda zyada
ko ka ki ke se tak mein
batao bataiye bolo boliye karo karna karne kijiye karunga karungi karenge chahiye chahta chahti
bhai ji yaar wala wali wale bhi toh aur lekin magar agar isliye
kuch koi sab sirf bas paisa kaam baat ghar
namaste dhanyavaad shukriya
milega milegi sakta sakti sakte lagta lagega
kar karke liye padega padta padti dena lena dijiye lijiye bhejo wapas chalo kyu kyo
mujhse tumse usko isko unko hamein hamen aisa aisi aise waisa jaise waise wahi yahi
paas saath sath andar bahar upar neeche dono
""".split())

_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']*")
_DEVA_CHAR = re.compile(r"[ऀ-ॣ॰-ॿ]")


def _latin_tokens(text: str) -> List[str]:
    return [t.lower().strip("'") for t in _LATIN_WORD.findall(text or "")]


def looks_hinglish(text: str) -> bool:
    """True when `text` is Hindi typed in Roman letters: mostly Latin script and rich in Hindi function words."""
    text = text or ""
    latin = len(re.findall(r"[A-Za-z]", text))
    non_latin = sum(1 for c in text if c.isalpha() and not c.isascii()) + len(_DEVA_CHAR.findall(text))
    letters = latin + non_latin
    if letters == 0 or latin / letters < 0.8:
        return False
    tokens = _latin_tokens(text)
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in HINGLISH_MARKERS)
    if hits >= 2 and hits / len(tokens) >= 0.25:
        return True
    return hits >= 1 and len(tokens) <= 2


def is_roman_script(text: str) -> bool:
    """True when most letters are Latin, i.e. the Hindi voice cannot read the text as it stands."""
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    other = sum(1 for c in (text or "") if c.isalpha() and not c.isascii())
    return latin > 0 and latin >= other


# ---------------------------------------------------------------------------------------------- dictionary

# "variant variant = देवनागरी" per line. Spellings follow how Indians say the word aloud, English loanwords included.
_DICT_SOURCE = """
hai = है
hain hein = हैं
ho = हो
hoon hun hu = हूँ
hoga = होगा
hogi = होगी
honge = होंगे
hua = हुआ
hui = हुई
hue = हुए
tha = था
thi = थी
thin = थीं
raha rha = रहा
rahi rhi = रही
rahe rhe = रहे
gaya gya = गया
gayi gayee gai = गई
gaye = गए
kiya = किया
kiye = किए
kar = कर
karo = करो
karna = करना
karni = करनी
karne = करने
karunga = करूँगा
karungi = करूँगी
karenge = करेंगे
karein karen = करें
kariye kijiye = कीजिए
karta = करता
karti = करती
karte = करते
kab = कब
dena = देना
dedo = दे दो
dijiye = दीजिए
diya = दिया
di = दी
diye = दिए
liya = लिया
lena = लेना
lo = लो
lijiye = लीजिए
aana = आना
aao = आओ
aaiye aiye = आइए
aaya = आया
aayi = आई
aaye = आए
aaunga = आऊँगा
aaungi = आऊँगी
aayega = आएगा
aayegi = आएगी
jana = जाना
jao = जाओ
jaiye = जाइए
jaunga = जाऊँगा
jaungi = जाऊँगी
jayega = जाएगा
jayegi = जाएगी
main mai = मैं
mujhe = मुझे
mujhko = मुझको
mera = मेरा
meri = मेरी
mere = मेरे
hum = हम
humein hume = हमें
humara hamara = हमारा
humari hamari = हमारी
humare hamare = हमारे
tum = तुम
tumhe tumhein = तुम्हें
tumhara = तुम्हारा
tumhari = तुम्हारी
tumhare = तुम्हारे
aap = आप
aapko = आपको
aapka = आपका
aapki = आपकी
aapke = आपके
aapse = आपसे
tu = तू
tera = तेरा
teri = तेरी
wo woh vo = वो
ye yeh = ये
yahan yaha = यहाँ
wahan vahan waha = वहाँ
kaun = कौन
kya = क्या
kyun kyon = क्यों
kyunki kyonki = क्योंकि
kaise = कैसे
kaisa = कैसा
kaisi = कैसी
kahan kahaan = कहाँ
kitna = कितना
kitni = कितनी
kitne = कितने
kaunsa = कौनसा
kis = किस
kisi = किसी
koi = कोई
kuch = कुछ
sab = सब
sabhi = सभी
sirf = सिर्फ़
bas = बस
bahut bohot bahot = बहुत
thoda = थोड़ा
thodi = थोड़ी
zyada jyada = ज़्यादा
kam = कम
aur = और
ya = या
lekin = लेकिन
magar = मगर
agar = अगर
isliye = इसलिए
toh = तो
to = तो
par = पर
pe = पे
me mein = में
ki = की
ke = के
ko = को
ka = का
se = से
tak = तक
nahi nahin nhi = नहीं
mat = मत
haan han = हाँ
ji jee = जी
accha acha achha achchha = अच्छा
theek thik theak = ठीक
sahi = सही
galat = गलत
bilkul = बिल्कुल
zaroor zarur jarur = ज़रूर
zaroori zaruri jaruri = ज़रूरी
shayad = शायद
pakka = पक्का
abhi = अभी
ab = अब
aaj = आज
kal = कल
parso = परसों
subah = सुबह
dopahar = दोपहर
shaam = शाम
raat = रात
din = दिन
hafta = हफ़्ता
mahina = महीना
saal = साल
baje = बजे
der = देर
jaldi = जल्दी
turant = तुरंत
pehle pahle = पहले
baad = बाद
phir fir = फिर
tab = तब
jab = जब
kabhi = कभी
hamesha = हमेशा
roz = रोज़
bhi = भी
namaste = नमस्ते
namaskar = नमस्कार
dhanyavaad dhanyawad dhanyavad = धन्यवाद
shukriya = शुक्रिया
maaf = माफ़
kripya = कृपया
bhai = भाई
behen bahan behan = बहन
dost = दोस्त
yaar = यार
beta = बेटा
beti = बेटी
sir = सर
madam = मैडम
saab sahab sahib = साहब
log = लोग
baat = बात
kaam = काम
paisa = पैसा
paise = पैसे
rupaye rupay = रुपये
ghar = घर
dukaan dukan = दुकान
gaadi gadi = गाड़ी
naam = नाम
pata = पता
jaankari = जानकारी
madad = मदद
samay = समय
waqt vakt = वक़्त
tarikh = तारीख
khana = खाना
paani pani = पानी
chai = चाय
sawal = सवाल
jawab = जवाब
dikkat = दिक्कत
pareshani = परेशानी
shikayat = शिकायत
batao = बताओ
bataiye = बताइए
batana = बताना
bata = बता
bolo = बोलो
boliye = बोलिए
bolna = बोलना
suno = सुनो
suniye = सुनिए
dekho = देखो
dekhiye = देखिए
dekhna = देखना
samajh = समझ
samjha samajha = समझा
samajhna = समझना
chahiye = चाहिए
chahta = चाहता
chahti = चाहती
chahte = चाहते
milega = मिलेगा
milegi = मिलेगी
milna = मिलना
rakho = रखो
rakhna = रखना
bhejo = भेजो
bhejna = भेजना
bhej = भेज
bhejiye = भेजिए
likho = लिखो
padho = पढ़ो
chalo = चलो
chalega = चलेगा
ruko = रुको
sakta = सकता
sakti = सकती
sakte = सकते
maloom malum = मालूम
lagta = लगता
lagega = लगेगा
hota = होता
hoti = होती
hote = होते
hona = होना
rehna = रहना
rehta = रहता
rehti = रहती
rehte = रहते
dikhao = दिखाओ
yaad = याद
mangna = माँगना
bura = बुरा
bada = बड़ा
badi = बड़ी
bade = बड़े
chhota = छोटा
chhoti = छोटी
naya = नया
nayi nayee = नई
purana = पुराना
sasta = सस्ता
mehnga mehenga = महँगा
tez = तेज़
dheere = धीरे
seedha sidha = सीधा
khali = खाली
poora pura = पूरा
aadha = आधा
pehla pahla = पहला
doosra dusra = दूसरा
teesra tisra = तीसरा
aakhri akhri = आखिरी
asaan aasan = आसान
mushkil = मुश्किल
mast = मस्त
badhiya = बढ़िया
zabardast = ज़बरदस्त
ek = एक
do = दो
teen = तीन
char chaar = चार
paanch panch = पाँच
chhe chhah = छह
saat = सात
aath = आठ
nau = नौ
das = दस
gyarah = ग्यारह
barah = बारह
bees = बीस
tees = तीस
pachaas = पचास
sau = सौ
hazaar hazar = हज़ार
lakh = लाख
crore karod = करोड़
somvar = सोमवार
mangalvar = मंगलवार
budhvar = बुधवार
guruvar veervar = गुरुवार
shukravar = शुक्रवार
shanivar = शनिवार
ravivar itwar = रविवार
yahin = यहीं
wahin vahin = वहीं
kahin = कहीं
ghanta = घंटा
ghante = घंटे
banta = बनता
bante = बनते
bana = बना
banao = बनाओ
banana = बनाना
sunna = सुनना
please pls plz = प्लीज़
sorry = सॉरी
thank = थैंक
thanks = थैंक्स
welcome = वेलकम
hello hallo = हैलो
hi = हाय
appointment = अपॉइंटमेंट
doctor dr = डॉक्टर
hospital = हॉस्पिटल
clinic = क्लिनिक
medicine = मेडिसिन
payment = पेमेंट
refund = रिफंड
order = ऑर्डर
booking = बुकिंग
book = बुक
cancel = कैंसल
confirm = कन्फर्म
form = फॉर्म
call = कॉल
price = प्राइस
subsidy = सब्सिडी
scheme = स्कीम
loan = लोन
bank = बैंक
account = अकाउंट
card = कार्ड
credit = क्रेडिट
debit = डेबिट
upi = यूपीआई
otp = ओटीपी
id = आईडी
aadhaar aadhar = आधार
pan = पैन
link = लिंक
app = ऐप
website = वेबसाइट
email = ईमेल
mail = मेल
message = मैसेज
whatsapp = व्हाट्सऐप
online = ऑनलाइन
offline = ऑफलाइन
delivery = डिलीवरी
deliver = डिलीवर
delivered = डिलीवर
product = प्रोडक्ट
service = सर्विस
customer = कस्टमर
support = सपोर्ट
help = हेल्प
issue = इश्यू
problem = प्रॉब्लम
complaint = कंप्लेंट
ticket = टिकट
train = ट्रेन
bus = बस
flight = फ्लाइट
hotel = होटल
room = रूम
table = टेबल
time = टाइम
date = डेट
schedule = शेड्यूल
slot = स्लॉट
office = ऑफिस
meeting = मीटिंग
manager = मैनेजर
team = टीम
sales = सेल्स
offer = ऑफर
discount = डिस्काउंट
cashback = कैशबैक
balance = बैलेंस
bill = बिल
invoice = इनवॉइस
receipt = रसीद
tax = टैक्स
gst = जीएसटी
insurance = इंश्योरेंस
policy = पॉलिसी
plan = प्लान
premium = प्रीमियम
document = डॉक्यूमेंट
certificate = सर्टिफिकेट
application = एप्लिकेशन
apply = अप्लाई
registration = रजिस्ट्रेशन
register = रजिस्टर
login = लॉगिन
password = पासवर्ड
number = नंबर
mobile = मोबाइल
phone = फ़ोन
internet = इंटरनेट
network = नेटवर्क
recharge = रिचार्ज
sim = सिम
data = डेटा
update = अपडेट
status = स्टेटस
check = चेक
verify = वेरिफाई
detail = डिटेल
details = डिटेल्स
information = इन्फॉर्मेशन
info = इन्फो
address = एड्रेस
pin = पिन
code = कोड
area = एरिया
city = सिटी
location = लोकेशन
store = स्टोर
shop = शॉप
market = मार्केट
ok okay = ओके
yes = यस
no = नो
sure = श्योर
minute = मिनट
minutes = मिनट
hour = घंटा
hours = घंटे
second = सेकंड
morning = मॉर्निंग
evening = इवनिंग
night = नाइट
weekend = वीकेंड
return = रिटर्न
exchange = एक्सचेंज
warranty = वारंटी
guarantee = गारंटी
replacement = रिप्लेसमेंट
size = साइज़
colour color = कलर
model = मॉडल
brand = ब्रांड
quality = क्वालिटी
stock = स्टॉक
available = अवेलेबल
urgent = अर्जेंट
emergency = इमरजेंसी
fee fees = फीस
admission = एडमिशन
course = कोर्स
class = क्लास
exam = एग्ज़ाम
result = रिज़ल्ट
school = स्कूल
college = कॉलेज
teacher = टीचर
student = स्टूडेंट
laptop = लैपटॉप
computer = कंप्यूटर
printer = प्रिंटर
free = फ्री
ready = रेडी
reply = रिप्लाई
send = सेंड
fine = फाइन
good = गुड
monday = मंडे
tuesday = ट्यूज़डे
wednesday = वेडनेसडे
thursday = थर्सडे
friday = फ्राइडे
saturday = सैटरडे
sunday = संडे
today = टुडे
tomorrow = टुमॉरो
usse = उससे
isse = इससे
unse = उनसे
dunga = दूँगा
dungi = दूँगी
lunga = लूँगा
lungi = लूँगी
kahiye kahiy = कहिए
kaho = कहो
kehna kahna = कहना
kehte kahte = कहते
liye = लिए
padega = पड़ेगा
padta = पड़ता
padti = पड़ती
padenge = पड़ेंगे
bharna = भरना
bhar = भर
wapas vapas = वापस
paas = पास
saath sath = साथ
andar = अंदर
bahar = बाहर
upar = ऊपर
neeche = नीचे
dono = दोनों
aisa = ऐसा
aisi = ऐसी
aise = ऐसे
waisa = वैसा
jaise = जैसे
waise = वैसे
wahi vahi = वही
yahi = यही
mujhse = मुझसे
tumse = तुमसे
usko = उसको
isko = इसको
unko = उनको
kyu kyo = क्यों
karke = करके
sharma = शर्मा
verma = वर्मा
gupta = गुप्ता
singh = सिंह
kumar = कुमार
patel = पटेल
yadav = यादव
mishra = मिश्रा
joshi = जोशी
khan = खान
reddy = रेड्डी
rahul = राहुल
amit = अमित
priya = प्रिया
ravi = रवि
ajay = अजय
vijay = विजय
sanjay = संजय
rohit = रोहित
neha = नेहा
pooja puja = पूजा
anita = अनिता
sunita = सुनिता
suresh = सुरेश
ramesh = रमेश
mahesh = महेश
rajesh = राजेश
deepak = दीपक
ankit = अंकित
ayush = आयुष
rohan = रोहन
sneha = स्नेहा
anjali = अंजलि
kavita = कविता
mohan = मोहन
ram = राम
krishna = कृष्णा
delhi = दिल्ली
mumbai = मुंबई
india = इंडिया
"""


def _build_dictionary() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in _DICT_SOURCE.strip().splitlines():
        keys, _, deva = line.partition("=")
        for k in keys.split():
            out[k.strip().lower()] = deva.strip()
    return out


DICTIONARY: Dict[str, str] = _build_dictionary()

# Words that are real English too: the dictionary still converts them when no model helps, but they never override a
# model's reading ("call me" must not become "कॉल में").
_NO_OVERRIDE = frozenset({"me", "main", "mai", "to", "hi", "do", "par", "pe", "is", "us", "no", "yes", "fine", "good",
                          "sir", "ok", "okay", "second", "pan", "pin", "id", "kar", "ab", "dr", "link", "return",
                          "order", "check", "free", "ready", "today", "tomorrow", "time", "date", "class", "update"})

_VOWEL_SIGNS = "ािीुूृेैोौंँ"


def _from_dictionary(word: str) -> Optional[str]:
    """Dictionary reading of one lower-case word, with a plural -s for English loanwords (doctors -> डॉक्टर्स)."""
    hit = DICTIONARY.get(word)
    if hit:
        return hit
    if len(word) > 3 and word.endswith("s") and word[:-1] in DICTIONARY:
        stem = DICTIONARY[word[:-1]]
        return stem + ("ज़" if stem[-1] in _VOWEL_SIGNS or stem[-1] in "आइईउऊएऐओऔ" else "्स")
    return None


# ---------------------------------------------------------------------------------------------- rule transliteration

_CONS = {"chh": "छ", "ksh": "क्ष", "kh": "ख", "gh": "घ", "ch": "च", "jh": "झ", "th": "थ", "dh": "ध", "ph": "फ",
         "bh": "भ", "sh": "श", "gy": "ज्ञ", "k": "क", "g": "ग", "j": "ज", "t": "त", "d": "द", "n": "न", "p": "प",
         "b": "ब", "m": "म", "y": "य", "r": "र", "l": "ल", "v": "व", "w": "व", "s": "स", "h": "ह", "f": "फ़",
         "z": "ज़", "q": "क़", "x": "क्स", "c": "क"}
_CKEYS = sorted(_CONS, key=len, reverse=True)
_VOWELS = {"aa": ("आ", "ा"), "ee": ("ई", "ी"), "ii": ("ई", "ी"), "oo": ("ऊ", "ू"), "uu": ("ऊ", "ू"),
           "ai": ("ऐ", "ै"), "au": ("औ", "ौ"), "ou": ("औ", "ौ"), "a": ("अ", ""), "e": ("ए", "े"), "i": ("इ", "ि"),
           "o": ("ओ", "ो"), "u": ("उ", "ु")}
_VKEYS = sorted(_VOWELS, key=len, reverse=True)
# A consonant followed by a consonant gets a virama only for real clusters; otherwise it is written plainly
# (कितना, मेहनत) and the reader supplies the schwa deletion Hindi speakers expect.
_CLUSTER_SECOND = {"r", "y", "v", "w"}  # (word-initial clusters, plan/blue, are handled separately)
_NO_CLUSTER_FIRST = {"n", "m", "h", "r", "y", "v", "w", "l"}
_CLUSTER_PAIRS = {("r", "y"), ("n", "y"), ("m", "h"), ("n", "h"), ("s", "t"), ("s", "th"), ("s", "k"), ("s", "p"),
                  ("s", "m"), ("s", "n"), ("s", "v"), ("s", "w"), ("k", "t"), ("b", "d"), ("k", "s"), ("t", "m"),
                  ("d", "m"), ("j", "n")}
_ENGLISH_SUFFIXES = (("tion", "शन"), ("sion", "शन"), ("ment", "मेंट"), ("ing", "इंग"))


def _tokenise(word: str) -> List[tuple]:
    toks, i = [], 0
    while i < len(word):
        c = next((k for k in _CKEYS if word.startswith(k, i)), None)
        if c:
            toks.append(("C", c))
            i += len(c)
            continue
        v = next((k for k in _VKEYS if word.startswith(k, i)), None)
        if v:
            toks.append(("V", v))
            i += len(v)
            continue
        i += 1  # stray character (apostrophe, digit): skip
    return toks


def _rule_word(word: str) -> str:
    """Hinglish-aware Roman -> Devanagari for a word the dictionary does not know."""
    for suffix, deva in _ENGLISH_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return _rule_word(word[:-len(suffix)]) + deva
    toks = _tokenise(word)
    out: List[str] = []
    n = len(toks)
    k = 0
    while k < n:
        kind, key = toks[k]
        nxt = toks[k + 1] if k + 1 < n else None
        prev_vowel = k > 0 and toks[k - 1][0] == "V"
        if kind == "V":
            out.append(_VOWELS[key][0])
            k += 1
            continue
        # consonant
        if key in ("n", "m") and prev_vowel and nxt and nxt[0] == "C" and nxt[1] not in ("h", "y", "r", "v", "w", "l",
                                                                                        "n", "m", "s", "sh"):
            out.append("ं")
            k += 1
            continue
        if key == "y" and nxt is None and k > 0 and toks[k - 1][0] == "C":
            out.append("ी")  # final -y after a consonant: baby, mummy
            k += 1
            continue
        deva = _CONS[key]
        if nxt is None:
            out.append(deva)
        elif nxt[0] == "V":
            vkey = nxt[1]
            is_last = k + 2 == n
            sign = _VOWELS[vkey][1]
            if is_last and vkey == "a":
                sign = "ा"
            elif is_last and vkey == "i":
                sign = "ी"
            elif is_last and vkey == "u" and k > 0:
                sign = "ू"
            out.append(deva + sign)
            k += 2
            # a vowel straight after a vowel is a new syllable: kaie, aaiye
            while k < n and toks[k][0] == "V":
                out.append(_VOWELS[toks[k][1]][0])
                k += 1
            continue
        else:  # consonant + consonant
            first, second = key, nxt[1]
            virama = (k == 0 or first == second
                      or (first, second) in _CLUSTER_PAIRS
                      or (second in _CLUSTER_SECOND and first not in _NO_CLUSTER_FIRST))
            out.append(deva + ("्" if virama else ""))
        k += 1
    return "".join(out)


def convert_word(word: str) -> str:
    """One Roman word to Devanagari: dictionary first, then the rules."""
    key = word.lower().strip("'")
    if not key:
        return word
    return _from_dictionary(key) or _rule_word(key)


def dictionary_to_devanagari(text: str) -> str:
    """Offline conversion of a whole text: every Roman word through the dictionary, else the rules."""
    return _LATIN_WORD.sub(lambda m: convert_word(m.group(0)), text or "")


# ---------------------------------------------------------------------------------------------- LLM conversion

_SYSTEM = ("You are a transliteration tool, not a chatbot. Rewrite the user's Hinglish (Hindi typed in Roman letters, "
           "often mixed with English words) in Devanagari script, word for word, exactly as it is spoken. Write "
           "English words in Devanagari the way Indians pronounce them. Keep numbers as digits. Never answer, "
           "translate, explain or add words. Output only the Devanagari text.")
_SHOTS = [("kal subah aana", "कल सुबह आना"),
          ("mera order abhi tak deliver nahi hua", "मेरा ऑर्डर अभी तक डिलीवर नहीं हुआ"),
          ("bhai price kitna hai?", "भाई प्राइस कितना है?"),
          ("aap kaise ho", "आप कैसे हो")]
# (marker in the model's answer, Roman words in the input that make the marker legitimate)
_REFUSALS = (("खेद", ("khed", "sorry")), ("माफ", ("maaf", "mafi", "sorry")), ("क्षमा", ("kshama", "sorry")),
             ("sorry", ("sorry",)), ("apolog", ("apolog",)), ("cannot", ("cannot",)), ("can't", ("can't",)),
             ("मैं एक", ()), ("language model", ()), ("as an ai", ()), ("मैं आपकी मदद", ()),
             ("मैं आपकी सहायता", ()), ("मैं इसमें", ()))
_ASCII_DIGITS = {0x0966 + i: ord(str(i)) for i in range(10)}
_PUNCT = ".,!?;:\"'()[]{}…-–—।“”‘’"
_CHUNK_WORDS = 60
_MAX_LLM_CHUNKS = 8


def _split_punct(tok: str):
    core = tok.strip(_PUNCT)
    if not core:
        return tok, "", ""
    start = tok.index(core)
    return tok[:start], core, tok[start + len(core):]


def _valid(src: str, out: str) -> bool:
    """Is `out` a Devanagari rewrite of `src` rather than a chatbot reply or a truncated answer?"""
    if not out:
        return False
    deva = len(_DEVA_CHAR.findall(out))
    latin = len(re.findall(r"[A-Za-z]", out))
    if deva == 0 or deva / (deva + latin) < 0.8:
        return False
    n, m = len(src.split()), len(out.split())
    if not (n * 0.5 <= m <= max(n * 1.5, n + 1)):
        return False
    low_src, low_out = src.lower(), out.lower()
    for marker, excused_by in _REFUSALS:
        if marker in low_out and not any(e in low_src for e in excused_by):
            return False
    return True


def _clean_llm_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text).strip()
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = text.translate(_ASCII_DIGITS)  # the model sometimes writes ५ for 5; numbers stay as digits
    if len(text) > 1 and text[0] in "\"“'‘" and text[-1] in "\"”'’":
        text = text[1:-1].strip()
    return text


def _repair(src: str, out: str) -> str:
    """Pin known words to the dictionary and replace any Latin the model left behind."""
    src_toks, out_toks = src.split(), out.split()
    aligned = len(src_toks) == len(out_toks)
    fixed = []
    for i, tok in enumerate(out_toks):
        lead, core, trail = _split_punct(tok)
        if not core:
            fixed.append(tok)
            continue
        if aligned:
            s_core = _split_punct(src_toks[i])[1]
            s_key = s_core.lower()
            known = None if s_key in _NO_OVERRIDE or not re.fullmatch(r"[a-z']+", s_key) else _from_dictionary(s_key)
            if known:
                core = known
            elif _LATIN_WORD.search(core) and re.fullmatch(r"[A-Za-z']+", s_core):
                core = convert_word(s_core)
        if _LATIN_WORD.search(core):  # still Latin: unaligned text or a mixed token the source could not repair
            if _DEVA_CHAR.search(core):
                core = _LATIN_WORD.sub("", core)
            else:
                core = _LATIN_WORD.sub(lambda m: convert_word(m.group(0)), core)
        fixed.append(lead + core + trail)
    return " ".join(fixed)


def _chunks(text: str) -> List[str]:
    """Split a long text at sentence ends into pieces of about _CHUNK_WORDS words."""
    if len(text.split()) <= _CHUNK_WORDS:
        return [text]
    out, cur, count = [], [], 0
    for sent in re.findall(r"[^.!?।\n]+[.!?।]*\s*", text):
        w = len(sent.split())
        if cur and count + w > _CHUNK_WORDS:
            out.append("".join(cur).strip())
            cur, count = [], 0
        cur.append(sent)
        count += w
    if cur:
        out.append("".join(cur).strip())
    return [c for c in out if c]


def _llm_chunk(chunk: str, timeout: float):
    """(text, provider) from the model, or None when its answer fails validation. Raises LLMUnavailable."""
    messages = [{"role": "system", "content": _SYSTEM}]
    for u, a in _SHOTS:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": chunk})
    words = len(chunk.split())
    res = llm.chat(messages, temperature=0.0, max_tokens=max(64, words * 10), timeout=timeout)
    out = _clean_llm_text(res.text)
    if not _valid(chunk, out):
        log.info("hinglish: model answer rejected for %r: %r", chunk[:60], out[:80])
        return None
    return _repair(chunk, out), res.provider


_cache: "OrderedDict[str, dict]" = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_SIZE = 512


def to_devanagari(text: str, *, use_llm: bool = True, timeout: float = 15.0) -> dict:
    """Roman Hinglish -> Devanagari. Returns {"text", "method": "llm"|"dictionary", "provider": str|None}.

    The model reads the sentence in context; its answer is validated and its known words pinned to the dictionary.
    Without a model, or when its answer is not a clean rewrite, the dictionary and rules convert the whole text."""
    text = (text or "").strip()
    if not text:
        return {"text": "", "method": "dictionary", "provider": None}
    with _cache_lock:
        hit = _cache.get(text)
        if hit is not None:
            _cache.move_to_end(text)
            return dict(hit)
    result = None
    if use_llm:
        pieces: List[str] = []
        providers: List[str] = []
        all_llm, llm_ok = True, True
        for i, chunk in enumerate(_chunks(text)):
            got = None
            if llm_ok and i < _MAX_LLM_CHUNKS:
                try:
                    got = _llm_chunk(chunk, timeout)
                except llm.LLMUnavailable as e:
                    log.info("hinglish: no language model (%s); using the dictionary", e)
                    llm_ok = False
                except Exception as e:  # noqa: BLE001
                    log.warning("hinglish: model call failed: %r", e)
                    llm_ok = False
            if got is None:
                all_llm = False
                pieces.append(dictionary_to_devanagari(chunk))
            else:
                pieces.append(got[0])
                providers.append(got[1])
        if providers:
            result = {"text": " ".join(pieces), "method": "llm" if all_llm else "dictionary",
                      "provider": providers[0] if all_llm else None}
    if result is None:
        result = {"text": dictionary_to_devanagari(text), "method": "dictionary", "provider": None}
    if result["method"] == "llm":
        with _cache_lock:
            _cache[text] = dict(result)
            while len(_cache) > _CACHE_SIZE:
                _cache.popitem(last=False)
    return result
