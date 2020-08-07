#include <iostream>
#include <string>

using namespace std;

bool verifica_bissexto(int ano) {
   if (((ano % 4 == 0) && (ano % 100 != 0)) || (ano % 400 == 0))
   return 1;
   else
   return 0;
}

string return_mes_extenso(int mes) {
  if(mes == 1) return "Janeiro";
  if(mes == 2) return "Fevereiro";
  if(mes == 3) return "Março";
  if(mes == 4) return "Abril";
  if(mes == 5) return "Maio";
  if(mes == 6) return "Junho";
  if(mes == 7) return "Julho";
  if(mes == 8) return "Agosto";
  if(mes == 9) return "Setembro";
  if(mes == 10) return "Outubro";
  if(mes == 11) return "Novembro";
  if(mes == 12) return "Dezembro";
  else return "Mes inexistente";
}

string return_signo(int dia, int mes) {
    string astro_sign = "";
    if (mes == 12) {
        if (dia < 22) return "Sagitario";
        else return "Capricornio";
    }
    else if (mes == 1) {
        if (dia < 20) return "Capricornio";
        else return "Aquario";
    }
    else if (mes == 2) {
        if (dia < 19) return "Aquario";
        else return "Peixes";
    }

    else if (mes == 3) {
        if (dia < 21) return "Peixes";
        else return "Aries";
    }
    else if (mes == 4) {
        if (dia < 20) return "Aries";
        else return "Touro";
    }

    else if (mes == 5) {
        if (dia < 21) return "Touro";
        else return "Gemeos";
    }

    else if (mes == 6) {
        if (dia < 21) return "Gemeos";
        else return "Cancer";
    }

    else if (mes == 7) {
        if (dia < 23) return "Cancer";
        else return "Leao";
    }

    else if (mes == 8) {
        if (dia < 23) return "Leao";
        else return "Virgem";
    }

    else if (mes == 9) {
        if (dia < 23) return "Virgem";
        else return "Libra";
    }

    else if (mes == 10) {
        if (dia < 23) return "Libra";
        else return "Escorpiao";
    }

    else if (mes == 11) {
        if (dia < 22) return "Escorpiao";
        else return "Sagitatio";
    }

    else return "erro";
}

class ClasseData {

  private:
    int dia, mes, ano;
    string signo, mes_extenso;
    bool bissexto;
  
  public:
    ClasseData(int dia, int mes, int ano);
    void calc_dados();
    void return_dados();

};

ClasseData::ClasseData(int dia, int mes, int ano) {
  this->dia = dia;
  this->mes = mes;
  this->ano = ano;
}

void ClasseData::calc_dados() {
  // verifica se é bissexto
  this->bissexto = verifica_bissexto(this->ano);
  this->mes_extenso = return_mes_extenso(this->mes);
  this->signo = return_signo(this->dia, this->mes);
}

void ClasseData::return_dados() {

  cout << this->dia << "/" << this->mes << "/" <<this->ano << "\n";

  cout << this->dia << " de " << this->mes_extenso << " de " << this->ano << "\n";

  if(this->bissexto) {
    cout << "E ano bissexto\n";
  }
  else {
    cout << "Nao e ano bissexto\n";
  }

  cout << "Signo: " << this->signo << "\n";
}


int main() {
  int dia, mes, ano;

  // Pede as entradas do usuário
  cout << "Digite a data de nascimento no formato dd/mm/aaaa: \n";
  scanf("%d/%d/%d", &dia, &mes, &ano);

  if(dia<1 || dia>31){
    // Cria um objeto da classe ClasseData
    ClasseData novaData(dia, mes, ano);

    // Faz os cálculos necessários
    novaData.calc_dados();

    // Retorna os resultados
    novaData.return_dados();
  }
}