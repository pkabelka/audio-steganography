# Audio Steganography

Tento repozitář obsahuje Python knihovnu pro kódování a dekódování libovolných
dat do krycího WAV audio souboru s použitím implementovaných steganografických
metod.

## Potřebné knihovny

- Python 3.8 nebo *novější*
- NumPy
- SciPy
- pandas (pro vyhodnocení metod)
- matplotlib (pro vyhodnocení metod)

Tyto knihovny je možné nainstalovat pomocí:

```
python -m pip install -r requirements.txt
```

## Použití programu

Vestavěný terminálový program je možné spustit následovně:

```
python -m audio_steganography
```

nebo pomocí:

```
./audio-steganography.sh
```

Program očekává název steganografické metody jako první argument následovaný
argumentem `encode` nebo `decode` na základě požadované operace.

### Kódování

Při kódování je nutné použít přepínač `-s` pro specifikaci krycího souboru a
poté je potřeba použít buď přepínač `-f` pro kódování souboru nebo `-t` pro
kódování textu z argumentu. Některé metody požadují zadání hodnoty nějakého
parametru. Navíc mají některé metody nepovinné parametry, které ovlivňují
výsledek kódování Tyto parametry je možné vypsat pomocí přepínače `-h` po
zadání názvu metody.

### Dekódování

Při dekódování je nutné použít přepínač `-s` pro specifikaci stego souboru.
Některé metody také vyžadují zadání některých parametrů, které byly použity při
kódování.

## Použití knihovny

Přidejte knihovnu do programu následovně:

```
import audio_steganography
```

a použijte metody z adresáře [methods](audio_steganography/methods).

Pokud chcete přidat vlastní metodu, pak [method_base
module](audio_steganography/methods/method_base.py) obsahuje abstraktní bázovou
třídu `MethodBase`, kterou musí dědit všechny steganografické metody. Poté musí
být nová metoda přidána do [MethoEnum
třídy](audio_steganography/methods/__init__.py) a všechny parametry metody musí
být přidány do slovníku `options` ve funkci
[`main`](audio_steganography/cli/__init__.py).

## LICENCE

Tento projekt je pod licencí Apache-2.0. Více informací se nachází v souborech
[LICENSE](LICENSE) a [NOTICE](NOTICE).
